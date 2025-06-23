from .dnse_utils import arbiter, setup_logger

"""
PortfolioManager allocates position sizes for multi-strategy systems

Key concepts:
--------------------------------
- Each strategy is uniquely identified by its class name ("strategy_name")
- Positions are signed integers: positive = LONG, negative = SHORT
- Before enterting a trade, strategy sends trade proposal to PortfolioManager to get approval
- PortfolioManager approves the trade proposal, blocks the approved budget, and marks position as pending
- Strategy instructs position manager to initiate order
- When order execution is done, position manager updates the realized position with PortfolioManager
- PortfolioManager unmarks the pending position
- No new approval is granted while the strategy's approved position is pending
- Exits do not require approval, but new realized positions still need to be finalized with PortfolioManager
"""
class PortfolioManager:
    _instance = None

    def __new__(cls):
        """Ensure singleton instance"""
        if cls._instance is None:
            cls._instance = super(PortfolioManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, max_gross_exposure=500, max_net_exposure=200, max_reward=0.5):
        if self._initialized:
            return

        self.logger = setup_logger(name=self.__class__.__name__, initial_log=False)
        self._active = False
        self.positions = {}  # {strategy_name: size}, positive = long, negative = short
        self.pendings = {} # {strategy_name: True/False}, True = pending, False = not pending
        self.max_gross_exposure = max_gross_exposure # maximum allowed gross position size
        self.max_net_exposure = max_net_exposure # maximum allowed directional bias
        self.max_reward = max_reward # maximum reward ratio for strategies that reduce net exposure
        self.position_history = [] # list of (strategy_name, size, reason)
        self.first_refresh = False # flag for one-time notification
        
        self._initialized = True
    
    def activate(self, max_gross_exposure=500, max_net_exposure=200, max_reward=0.5):
        self._active = True
        self.positions = {}
        self.pendings = {}
        self.max_gross_exposure = max_gross_exposure
        self.max_net_exposure = max_net_exposure
        self.max_reward = max_reward
        self.logger.info("PortfolioManager activated")

    def deactivate(self):
        self._active = False
        self.positions = {}
        self.pendings = {}
        self.logger.info("PortfolioManager deactivated")

    def register_strategy(self, strategy):
        if not self._active:
            return
        strategy_name = strategy.__class__.__name__
        if strategy_name not in self.positions:
            self.positions[strategy_name] = 0
            self.pendings[strategy_name] = False
        #self.logger.info(f"Registered {strategy_name}. Status: {self.to_dict()}")
    
    def refresh_params(self):
        """Refresh parameters in live trading, with values from arbiter"""
        if not self._active:
            return
                
        self.max_gross_exposure = arbiter.get_param(self, "max_gross_exposure", int, self.max_gross_exposure)
        self.max_net_exposure = arbiter.get_param(self, "max_net_exposure", int, self.max_net_exposure)
        self.max_reward = arbiter.get_param(self, "max_reward", float, self.max_reward)
        
        if not self.first_refresh:
            self.first_refresh = True
            self.logger.info(f"PortfolioManager first refresh. Status: {self.to_dict()}")

    def finalize_position(self, strategy, size, reason=None):
        if not self._active:
            return
        strategy_name = strategy.__class__.__name__
        self.positions[strategy_name] = size
        self.pendings[strategy_name] = False

        history_entry = {
            'date': strategy.data.datetime.datetime().date(),
            'time': strategy.data.datetime.datetime().time(),
            'strategy_name': strategy_name,
            'size': size,
            'reason': reason
        }
        history_entry.update(self.to_dict())
        self.position_history.append(history_entry)

        if strategy.data.isinlivemode():
            self.logger.info(f"finalize_position: strategy={strategy_name}, size={size}, reason={reason}, status={self.to_dict()}")
            
    def request_approval(self, strategy, trade_size, multiplier, trade_direction, max_reward=None):
        if not self._active:  # approve as-is in inactive mode
            return int(trade_size * multiplier)
        
        strategy_name = strategy.__class__.__name__
        assert strategy_name in self.positions, f"Strategy {strategy_name} not registered"
        assert strategy_name in self.pendings, f"Strategy {strategy_name} not in pendings"
        assert trade_direction.upper() in ["LONG", "SHORT"], "trade_direction must be 'LONG' or 'SHORT'"

        # simple mechanism for now: just apply multiplier to trade size
        trade_size = int(trade_size * multiplier)

        if self.pendings[strategy_name]: # do not approve if pending
            if strategy.data.isinlivemode():
                self.logger.info(f"Pending position for {strategy_name}, do not approve. Status: {self.to_dict()}")
            return 0
        
        current_gross = sum(abs(size) for size in self.positions.values())
        current_net = sum(self.positions.values())
        trade_sign = 1 if trade_direction.upper() == "LONG" else -1

        # reward strategy that reduces net exposure with bigger trade size
        max_reward = min(max_reward, self.max_reward) if max_reward is not None else self.max_reward
        if abs(current_net) < 1e-6: # currently zero net exposure
            reward_factor = 1
        else:
            new_net = current_net + trade_size * trade_sign
            new_net_ratio = abs(new_net) / abs(current_net)
            if new_net_ratio < 1:
                reward_factor = 1 + (1 - new_net_ratio) * max_reward # max factor = 1 + max_reward
            else:
                reward_factor = 1
        trade_size = int(reward_factor * trade_size)

        gross_room = self.max_gross_exposure - (current_gross - abs(self.positions[strategy_name]))        
        net_room = max(0, self.max_net_exposure - current_net * trade_sign)
        approved_size = max(0, min(trade_size, gross_room, net_room))

        if approved_size > 0:
            self.positions[strategy_name] += approved_size * trade_sign
            self.pendings[strategy_name] = True # approved some budget -> block new approval
            if strategy.data.isinlivemode():
                self.logger.info(f"Approved for {strategy_name}: {approved_size} {trade_direction} trade. Position: ({self.positions[strategy_name]}, {self.pendings[strategy_name]})")
        else:
            self.pendings[strategy_name] = False # no budget approved -> do not block new approval
            if strategy.data.isinlivemode():
                self.logger.info(f"No budget approved for {strategy_name}. Position: ({self.positions[strategy_name]}, {self.pendings[strategy_name]})")
        
        return approved_size

    def to_dict(self):
        if not self._active:
            return {}
        out = {}
        for strategy_name, size in self.positions.items():
            out[f"pm_{strategy_name}_size"] = size
            out[f"pm_{strategy_name}_pending"] = self.pendings[strategy_name]
        out['pm_gross'] = sum(abs(size) for size in self.positions.values())
        out['pm_net'] = sum(self.positions.values())
        out["pm_max_gross"] = self.max_gross_exposure
        out["pm_max_net"] = self.max_net_exposure
        out["pm_max_reward"] = self.max_reward
        return out
    
    def get_current_gross(self):
        if not self._active:
            return 0
        return sum(abs(size) for size in self.positions.values())
    
    def get_current_net(self):
        if not self._active:
            return 0
        return sum(self.positions.values())

# Global singleton instance; must activate() to use
portfolio_manager = PortfolioManager()

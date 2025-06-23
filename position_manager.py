from datetime import timedelta, time, date
import random
import backtrader as bt
from .dnse_utils import now_in_vietnam_time, gsheet_logger, arbiter
from .portfolio_manager import portfolio_manager

"""
PositionManager handles position tracking, order updates, and SL/TP management for individual strategies.

Key concepts:
--------------------------------
Orders:
- Only created via controlled functions while there are no pending orders
- Custom information in kwargs is added to the orders
- No zero-size orders 
- All pending orders have the same sign (long or short)
- has_pending_orders() discards expired/inactive orders before checking for pending orders

Position actions:
- enter_long: Create new long position or increase existing long position
- enter_short: Create new short position or increase existing short position  
- exit_position: 
    - Partial or full exit; size is capped so there will be no position flip
    - Forced liquidation (attempt full exit then reset remaining position to zero)

Logging:
- New or increase position: log the incremental change, zero pnl
- Partial or full exit: log the incremental change and its corresponding pnl
- Reset position: log the reset with zero price, pnl, commission
- Capture custom information from executed orders

Interface with arbiter and portfolio manager:
- Enter position: request size from arbiter, request approval from portfolio manager
- Size change, entering position but no order created, zero-execution orders: finalize position with portfolio manager
"""
class PositionManager:
    
    (LONG_SL, LONG_TP, SHORT_SL, SHORT_TP) = range(4)

    SL_TP_REASONS = {
        LONG_SL: "LONG_SL",
        LONG_TP: "LONG_TP", 
        SHORT_SL: "SHORT_SL",
        SHORT_TP: "SHORT_TP"
    }

    circuit_breaker_datetime = None # datetime of circuit-breaker trigger

    def __init__(self, strategy):
        self.strategy = strategy
        self.logger = strategy.logger

        # Position state
        self.size = 0  # Current position size (+ for long, - for short)
        self.price = 0.0  # Average entry price
        self.entry_commission = 0.0  # Commission for entry
        
        # SL/TP
        self.auto_sl_tp = False  # Disabled by default for safety
        self.sl_ratio = None  # Stop loss ratio, e.g. 0.01
        self.tp_ratio = None  # Take profit ratio, e.g. 0.03
        self.trailing_stop = False
        self.sl_price = None
        self.tp_price = None
        self.best_price = None # Tracking best price for trailing stop

        # Pending orders
        self._pending_orders = {} # {order.ref -> order}

        # Forced liquidation flag
        self.force_liquidation = False

        # Execution algorithm
        self.min_slice_size = 10
        self.max_slice_size = 40
        self.regular_order_expiration_minutes = 16
        self.sl_tp_order_expiration_minutes = 2
        self.min_order_submission_delay = 2 # seconds
        self.max_order_submission_delay = 10 # seconds

        # Transaction log
        self.transaction_log = []
        self.trade_exit_log = []

        # register strategy with portfolio manager
        portfolio_manager.register_strategy(self.strategy)

    def enable_sl_tp(self, sl_ratio=0.01, tp_ratio=0.03, trailing_stop=False):
        """
        Enable automatic stop loss and take profit management.
        
        Args:
            sl_ratio (float): Stop loss ratio from entry price (e.g., 0.01 for 1%)
            tp_ratio (float): Take profit ratio from entry price (e.g., 0.03 for 3%)
            trailing_stop (bool): Whether to enable trailing stop
        """
        self.auto_sl_tp = True
        self.sl_ratio = sl_ratio
        self.tp_ratio = tp_ratio
        self.trailing_stop = trailing_stop

    def disable_sl_tp(self):
        self.auto_sl_tp = False
        self.sl_ratio = None
        self.tp_ratio = None
        self.trailing_stop = False
    
    def _update_sl_tp_prices(self):
        """Update SL/TP prices based on current price and position size"""
        if self.size == 0: # reset prices if position is neutral
            self.sl_price = None
            self.tp_price = None
            self.best_price = None
            return
        
        if self.auto_sl_tp:
            if self.size > 0: # LONG
                self.sl_price = self.price * (1 - self.sl_ratio)
                self.tp_price = self.price * (1 + self.tp_ratio)
            else: # SHORT
                self.sl_price = self.price * (1 + self.sl_ratio)
                self.tp_price = self.price * (1 - self.tp_ratio)
    
    def _has_pending_orders(self):
        """
        Returns True if there are still active pending orders
        Cleans up:
        - Orders older than 12 hours (real time, not bar time)
        - Orders past their custom expiry
        - Orders in terminal status
        """
        now = now_in_vietnam_time()
        cleaned_orders = {}

        for ref, order in self._pending_orders.items():
            try:
                # Skip orders that are too old (e.g., zombie orders)
                created = order.info.get('_createdDate', None)
                if not created or now - created > timedelta(hours=12):
                    continue

                # Cancel orders past their expiry window
                expiry = order.info.get('_expiry', None)
                if expiry and now > expiry:
                    self.strategy.broker.cancel_order(order)
                    continue

                # Keep only active orders
                if order.status in [order.Submitted, order.Accepted, order.Partial]:
                    cleaned_orders[ref] = order

            except Exception as e:
                self.logger.info(f"Error while validating pending order {ref}: {e}")
                continue

        self._pending_orders = cleaned_orders
        return len(self._pending_orders) > 0
        
    def _log_transaction(self, transaction_type, reason, size, entry_price=0.0, exit_price=0.0, pnl=0.0, entry_commission=0.0, exit_commission=0.0, order=None):
        commission = entry_commission + exit_commission
        pnlcomm = pnl - commission
        transaction = {
            'strategy': self.strategy.__class__.__name__,
            'type': transaction_type,
            'reason': reason,
            'date': self.strategy.data.datetime.datetime().date(),
            'time': self.strategy.data.datetime.datetime().time(),
            'day': self.strategy.data.datetime.datetime().strftime('%a'),
            'year': self.strategy.data.datetime.datetime().year,
            'size': size,
            'size_abs': abs(size),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'sl_price': self.sl_price,
            'tp_price': self.tp_price,
            'trailing_stop': self.trailing_stop,
            'pnl': pnl,
            'entry_commission': entry_commission,
            'exit_commission': exit_commission,
            'commission': commission,
            'pnlcomm': pnlcomm,
            'value': self.strategy.broker.getvalue(),
            'end_size': self.size,
            'pm_gross': portfolio_manager.get_current_gross(),
            'pm_net': portfolio_manager.get_current_net()
        }
        
        # Add custom info from order, skipping internal keys that start with _
        if order is not None and hasattr(order, 'info') and order.info:
            public_info = {k: v for k, v in order.info.items() if not k.startswith('_')}
            if public_info:
                transaction.update(public_info)

        transaction.update(portfolio_manager.to_dict())
        self.transaction_log.append(transaction)

        # log to exit_log
        if transaction_type == 'EXIT_POSITION':
            trade_exit = {
                'strategy': self.strategy.__class__.__name__,
                'type': transaction_type,
                'reason': reason,
                'date': self.strategy.data.datetime.datetime().date(),
                'time': self.strategy.data.datetime.datetime().time(),
                'day': self.strategy.data.datetime.datetime().strftime('%a'),
                'year': self.strategy.data.datetime.datetime().year,
                'size': size,
                'size_abs': abs(size),
                'entry_price': entry_price,
                'exit_price': exit_price,
                'price_diff': exit_price - entry_price,
                'pnl': pnl,
                'pnl_win': 1 if pnl > 0 else 0,
                'entry_commission': entry_commission,
                'exit_commission': exit_commission,
                'commission': commission,
                'pnlcomm': pnlcomm,
                'pnlcomm_win': 1 if pnlcomm > 0 else 0,
                'pnlcomm_avg_points': pnlcomm / (abs(size) * 100_000),
                'value': self.strategy.broker.getvalue(),
                'end_size': self.size,
                'pm_gross': portfolio_manager.get_current_gross(),
                'pm_net': portfolio_manager.get_current_net()
            }
            if order is not None and hasattr(order, 'info') and order.info:
                public_info = {k: v for k, v in order.info.items() if not k.startswith('_')}
                if public_info:
                    trade_exit.update(public_info)

            trade_exit.update(portfolio_manager.to_dict())
            trade_exit['pnl_'] = pnl  # for convenience when doing manual analysis: add pnl again at end
            self.trade_exit_log.append(trade_exit)

        # log to gsheet
        if transaction_type == 'EXIT_POSITION':
            side = "BUY" if size > 0 else "SELL"
            row_data = [
                self.strategy.__class__.__name__, # strategy name
                reason, # reason
                order.ref if order is not None else "No order", # order ref
                side, # side
                abs(size), # size
                entry_price, # entry price
                exit_price, # exit price
                (exit_price - entry_price) if side == "SELL" else (entry_price - exit_price), # exit price difference
                pnl, # pnl
                commission, # commission
                pnlcomm, # pnlcomm
                "Yes" if pnlcomm > 0 else "No", # profitable
                self.size, # remaining position
                portfolio_manager.get_current_gross(), # pm_gross
                portfolio_manager.get_current_net() # pm_net
            ]
            gsheet_logger.log_message("_exit", row_data)

    def _finalize_position_with_portfolio_manager(self, reason=None):
        """Finalize position, update the portfolio manager if no pending orders"""
        if not self._has_pending_orders():
            portfolio_manager.finalize_position(self.strategy, self.size, reason)

    def _set_position(self, size, price, entry_commission, reason=None):
        """
        Set position size and price, update the portfolio manager if no pending orders. 
        All changes in size, price, or entry commission should be done via this function.
        """
        self.size = size
        self.price = price
        self.entry_commission = entry_commission
        self._finalize_position_with_portfolio_manager(reason)
        self._update_sl_tp_prices()
                      
    def _enter_position(self, order):
        new_size = self.size + order.executed.size
        new_price = (self.price * self.size + order.executed.price * order.executed.size) / new_size
        new_entry_commission = self.entry_commission + order.executed.comm
        reason = order.info.get('reason') if hasattr(order, 'info') and order.info else None
        self._set_position(new_size, new_price, new_entry_commission, reason)

        self._log_transaction(
            transaction_type='ENTER_POSITION',
            reason=reason,
            size=order.executed.size,
            entry_price=order.executed.price,
            entry_commission=order.executed.comm,
            order=order
        )

    def _exit_position(self, order):
        new_size = self.size + order.executed.size
        taken_entry_commission = self.entry_commission * abs(order.executed.size / self.size) if self.size != 0 else 0.0
        new_entry_commission = self.entry_commission - taken_entry_commission
        reason = order.info.get('reason') if hasattr(order, 'info') and order.info else None
        self._set_position(new_size, self.price, new_entry_commission, reason)

        # log transaction
        pnl = (self.price - order.executed.price) * order.executed.size * 100_000
        self._log_transaction(
            transaction_type='EXIT_POSITION',
            reason=reason,
            size=order.executed.size,
            entry_price=self.price,
            exit_price=order.executed.price,
            pnl=pnl,
            entry_commission=taken_entry_commission,
            exit_commission=order.executed.comm,
            order=order
        )

        if self.force_liquidation and self.size != 0 and not self._has_pending_orders():
            liquidated_size = self.size
            reason = "FORCE_LIQUIDATION"
            self._set_position(size=0, price=0.0, entry_commission=0.0, reason=reason) # reset position
            self.force_liquidation = False

            self._log_transaction(
                transaction_type='EXIT_POSITION',
                reason=reason,
                size=liquidated_size
            )

    def update(self, order):
        """Update position state based on order execution"""

        # Only process orders in pending_orders (e.g. belong to this position manager)
        if order.ref not in self._pending_orders:
            return
        
        # log to gsheet accepted orders
        if order.status == order.Accepted:
            gsheet_logger.log_order(order)

        # Update pending orders
        self._pending_orders[order.ref] = order

        # Not process if order still pending
        if order.status in [order.Submitted, order.Accepted, order.Partial]:
            return
        
        # Process all orders that reached final status (e.g. Canceled, Expired, Rejected, Margin)
        # - full or partially filled: enter or exit position
        # - zero-size execution: finalize position with portfolio manager
        if order.executed.size == 0:
            self._finalize_position_with_portfolio_manager("ZERO_EXECUTION")
        elif (self.size == 0 or self.size * order.executed.size > 0): # enter: from neutral or increase long/short position
            self._enter_position(order)
        else: # size and executed.size have opposite signs: exit position
            self._exit_position(order)

        # log to gsheet
        gsheet_logger.log_order(order, portfolio_manager.get_current_gross(), portfolio_manager.get_current_net())

    def _circuit_breaker(self):
        # circuit-breaker only applies in live mode
        if not self.strategy.data.isinlivemode():
            return False

        # Only apply for 1-minute timeframe
        if not (hasattr(self.strategy.data, 'timeframe') and 
                self.strategy.data.timeframe == bt.TimeFrame.Minutes and
                self.strategy.data.compression == 1):
            return False
        
        # circuit-break if bar HL greater than limit
        HL_limit = 20 # NOTE: threshold found from 5 years of data; triggered 4 times in 5 years
        if self.strategy.data.high[0] - self.strategy.data.low[0] > HL_limit:
            self.circuit_breaker_datetime = self.strategy.data.datetime.datetime()
            gsheet_logger.log_info(f"{self.strategy.__class__.__name__} circuit-breaker triggered at {self.circuit_breaker_datetime}")

        # active for rest of the day
        return self.circuit_breaker_datetime is not None and self.strategy.data.datetime.date() == self.circuit_breaker_datetime.date()
    
    def _create_order(self, size, is_long, exectype=bt.Order.Market, reason=None, split_orders=False, **kwargs):
        """Create orders, return created size (0 if no order created)"""
        assert size >= 0, "Size must be positive"
        if size == 0 or self._has_pending_orders() or self._circuit_breaker():
            return 0
                
        # create slices of size between min_slice_size and max_slice_size
        slices = []
        if split_orders or self.strategy.data.isinlivemode(): # always split orders in live mode
            remaining_size = size  
            while remaining_size > 0:
                slice = min(remaining_size, random.randint(self.min_slice_size, self.max_slice_size))
                slices.append(slice)
                remaining_size -= slice
        else:
            slices.append(size)

        created_size = 0
        expiry = self.sl_tp_order_expiration_minutes if reason in self.SL_TP_REASONS.values() else self.regular_order_expiration_minutes
        for slice in slices:
            kwargs['_createdDate'] = now_in_vietnam_time() # used internally by position manager
            kwargs['_expiry'] = now_in_vietnam_time() + timedelta(minutes=expiry) # used internally by position manager
            kwargs['submission_delay'] = random.uniform(self.min_order_submission_delay, self.max_order_submission_delay) # seconds; info passed to store
            kwargs['expected_price'] = self.strategy.data.close[0] # expected price = price at decision time, to calculate slippage
            if reason:
                kwargs['reason'] = reason
            order = self.strategy.buy(size=slice, exectype=exectype, **kwargs) if is_long else \
                    self.strategy.sell(size=slice, exectype=exectype, **kwargs)
            if order: # order object is created
                self._pending_orders[order.ref] = order
                created_size += slice
        return created_size

    def enter_position(self, size, multiplier, trade_direction, exectype=bt.Order.Market, reason=None, split_orders=False, **kwargs):
        assert size >= 0, "Size must be positive"
        assert multiplier >= 0, "Multiplier must be positive"
        assert trade_direction.upper() in ["LONG", "SHORT"], "Trade direction must be LONG or SHORT"

        # get base size from arbiter, then request approval from portfolio manager
        size = arbiter.get_param(self.strategy, "size", int, size)
        size = portfolio_manager.request_approval(self.strategy, size, multiplier, trade_direction)
        
        if trade_direction.upper() == "LONG":
            created_size = self._create_order(size, is_long=True, exectype=exectype, reason=reason or "LONG_ENTRY", split_orders=split_orders, **kwargs)
        else:
            created_size = self._create_order(size, is_long=False, exectype=exectype, reason=reason or "SHORT_ENTRY", split_orders=split_orders, **kwargs)
        
        if created_size == 0: # no order created => immediate finalize position
            self._finalize_position_with_portfolio_manager("NO_ORDER_CREATED")
        
        return created_size

    def exit_position(self, size, exectype=bt.Order.Market, reason="EXIT_POSITION", split_orders=False, force_liquidation=False, **kwargs):
        """Exit position. Convention: size > 0"""
        size = min(abs(size), abs(self.size)) # size is capped by current position size, ensuring no position flip
        self.force_liquidation = force_liquidation
        if force_liquidation:
            size = abs(self.size)
        return self._create_order(size, is_long=self.size<0, exectype=exectype, reason=reason, split_orders=split_orders, **kwargs)

    def close_position(self, exectype=bt.Order.Market, reason="CLOSE_POSITION", split_orders=False, force_liquidation=False, **kwargs):
        """Full exit"""
        size = abs(self.size)
        return self.exit_position(size=size, exectype=exectype, reason=reason, split_orders=split_orders, force_liquidation=force_liquidation, **kwargs)
    
    def check_sl_tp(self, current_price=None):
        """Check if SL/TP conditions, return reason (str) if conditions are met, else None"""
        # Don't manage if:
        # - SL/TP is disabled
        # - No position
        # - Has pending orders (avoid race condition)
        if not self.auto_sl_tp or self.size == 0 or self._has_pending_orders():
            return None
        
        # Don't manage in ATO and ATC sessions
        # ATO = when most recent bar time is 14h45 of previous day, or 8h45 of current day in simulation mode
        # ATC = when most recent bar time is 14h30 (before KRX) or 14h29 (after KRX)
        bar_time = self.strategy.data.datetime.time()
        if bar_time == time(14, 45) or bar_time == time(8, 45) or bar_time == time(14, 30) or bar_time == time(14, 29):
            return None
        
        current_price = current_price or self.strategy.data.close[0]
        
        # Update best price and trailing stop if enabled
        if self.trailing_stop:
            if self.best_price is None:
                self.best_price = self.price # initialize on first update
            
            if self.size > 0:
                if current_price > self.best_price:
                    self.best_price = current_price
                    self.sl_price = self.best_price * (1 - self.sl_ratio)
            
            elif self.size < 0:
                if current_price < self.best_price:
                    self.best_price = current_price
                    self.sl_price = self.best_price * (1 + self.sl_ratio)
                    
        # Check SL/TP conditions
        reason = None
        if self.size > 0:
            if current_price < self.sl_price:
                reason = self.SL_TP_REASONS[self.LONG_SL]
            elif current_price > self.tp_price:
                reason = self.SL_TP_REASONS[self.LONG_TP]
        elif self.size < 0:
            if current_price > self.sl_price:
                reason = self.SL_TP_REASONS[self.SHORT_SL]
            elif current_price < self.tp_price:
                reason = self.SL_TP_REASONS[self.SHORT_TP]
        return reason
    
    def execute_sl_tp(self, reason, split_orders=False, **kwargs):
        """Execute SL/TP, should be called only after check_sl_tp() returns a reason"""
        return self.close_position(reason=reason, split_orders=split_orders, **kwargs)
    
    def manage_sl_tp(self, current_price=None, split_orders=False, **kwargs):
        """Check and execute SL/TP (stop loss and take profit) and trailing stop; return True if executed, else False"""
        reason = self.check_sl_tp(current_price)
        return self.execute_sl_tp(reason=reason, split_orders=split_orders, **kwargs) if reason else False

    def force_position(self, size, price, reason="FORCE_POSITION"):
        """
        Force a position with specified size and price. Condition: current position must 
        be neutral (size = 0), else return False.
        
        Typically used when a strategy needs to start trading from an existing position,
        such as:
        - Loading saved positions from previous sessions
        - Taking over positions from another strategy
        - Reconciling with actual broker positions
        
        Args:
            size (int): Target position size. Positive for long, negative for short
            price (float): Position entry price
            reason (str, optional): Reason for the position force
        """
        if size == 0 or self.size != 0:
            return False
                
        self._set_position(size, price, 0.0, reason) # zero entry commission

        self._log_transaction(
            transaction_type='ENTER_POSITION',
            reason=reason,
            size=size,
            entry_price=price
        )
        return True
    
    def force_price(self, new_price, reason="FORCE_PRICE"):
        """
        Force update the position's price and record the resulting P&L, while keeping position size unchanged.
        Typically used for mark-to-market adjustments, position reconciliation, or price corrections.

        Args:
            new_price (float): The new price to mark the position to
            reason (str, optional): Description of why the price is being updated
        """
        if self.size == 0 or self.price == new_price:
            return False
                
        pnl = (new_price - self.price) * self.size * 100_000
        self._set_position(self.size, new_price, 0.0, reason) # zero entry commission

        self._log_transaction(
            transaction_type='EXIT_POSITION', # type is EXIT_POSITION with size unchanged and non-zero pnl
            reason=reason,
            size=0,
            exit_price=new_price,
            pnl=pnl
        )
        return True

import argparse
import os
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional

import requests
from dotenv import load_dotenv
from web3 import Web3

ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [
            {"name": "owner", "type": "address"},
            {"name": "spender", "type": "address"},
        ],
        "name": "allowance",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": False,
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function",
    },
]

ROUTER_ABI = [
    {
        "name": "getAmountsOut",
        "outputs": [{"name": "", "type": "uint256[]"}],
        "inputs": [
            {"name": "amountIn", "type": "uint256"},
            {"name": "path", "type": "address[]"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "name": "swapExactTokensForTokens",
        "outputs": [{"name": "", "type": "uint256[]"}],
        "inputs": [
            {"name": "amountIn", "type": "uint256"},
            {"name": "amountOutMin", "type": "uint256"},
            {"name": "path", "type": "address[]"},
            {"name": "to", "type": "address"},
            {"name": "deadline", "type": "uint256"},
        ],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "name": "swapExactTokensForTokensSupportingFeeOnTransferTokens",
        "outputs": [],
        "inputs": [
            {"name": "amountIn", "type": "uint256"},
            {"name": "amountOutMin", "type": "uint256"},
            {"name": "path", "type": "address[]"},
            {"name": "to", "type": "address"},
            {"name": "deadline", "type": "uint256"},
        ],
        "stateMutability": "nonpayable",
        "type": "function",
    },
]


def bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class AgentConfig:
    rpc_url: str
    private_key: str
    chain_id: int
    token_address: str
    router_address: str
    sell_path: List[str]
    gecko_network: str
    sell_percent: Decimal
    take_profit_usd: Decimal
    stop_loss_usd: Decimal
    slippage_bps: int
    check_interval_seconds: int
    sell_once: bool
    dry_run: bool
    gas_limit: int
    gas_price_gwei: Decimal
    use_fee_on_transfer: bool

    @classmethod
    def from_env(cls) -> "AgentConfig":
        rpc_url = os.getenv("RPC_URL", "").strip()
        private_key = os.getenv("PRIVATE_KEY", "").strip()
        token_address = os.getenv("TOKEN_ADDRESS", "").strip()
        router_address = os.getenv("ROUTER_ADDRESS", "").strip()
        sell_path_raw = os.getenv("SELL_PATH", "").strip()
        gecko_network = os.getenv("GECKO_NETWORK", "eth").strip()

        if not all([rpc_url, private_key, token_address, router_address, sell_path_raw]):
            raise ValueError(
                "Missing required env vars. Set RPC_URL, PRIVATE_KEY, TOKEN_ADDRESS, "
                "ROUTER_ADDRESS, and SELL_PATH."
            )

        sell_path = [p.strip() for p in sell_path_raw.split(",") if p.strip()]
        if len(sell_path) < 2:
            raise ValueError("SELL_PATH must contain at least 2 token addresses.")

        return cls(
            rpc_url=rpc_url,
            private_key=private_key,
            chain_id=int(os.getenv("CHAIN_ID", "1")),
            token_address=token_address,
            router_address=router_address,
            sell_path=sell_path,
            gecko_network=gecko_network,
            sell_percent=Decimal(os.getenv("SELL_PERCENT", "1.0")),
            take_profit_usd=Decimal(os.getenv("TAKE_PROFIT_USD", "0")),
            stop_loss_usd=Decimal(os.getenv("STOP_LOSS_USD", "0")),
            slippage_bps=int(os.getenv("SLIPPAGE_BPS", "300")),
            check_interval_seconds=int(os.getenv("CHECK_INTERVAL_SECONDS", "30")),
            sell_once=bool_env("SELL_ONCE", True),
            dry_run=bool_env("DRY_RUN", True),
            gas_limit=int(os.getenv("GAS_LIMIT", "400000")),
            gas_price_gwei=Decimal(os.getenv("GAS_PRICE_GWEI", "0")),
            use_fee_on_transfer=bool_env("USE_FEE_ON_TRANSFER", False),
        )


class SellAgent:
    def __init__(self, cfg: AgentConfig) -> None:
        self.cfg = cfg
        self.web3 = Web3(Web3.HTTPProvider(cfg.rpc_url))
        if not self.web3.is_connected():
            raise ConnectionError("Failed to connect to RPC_URL.")

        self.account = self.web3.eth.account.from_key(cfg.private_key)
        self.wallet_address = self.account.address

        self.token = self.web3.eth.contract(
            address=Web3.to_checksum_address(cfg.token_address),
            abi=ERC20_ABI,
        )
        self.router = self.web3.eth.contract(
            address=Web3.to_checksum_address(cfg.router_address),
            abi=ROUTER_ABI,
        )
        self.path = [Web3.to_checksum_address(addr) for addr in cfg.sell_path]
        self.decimals = self.token.functions.decimals().call()

    def get_token_price_usd(self) -> Optional[Decimal]:
        url = (
            "https://api.geckoterminal.com/api/v2/networks/"
            f"{self.cfg.gecko_network}/tokens/{self.cfg.token_address}"
        )
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None
        payload = response.json()
        price = (
            payload.get("data", {})
            .get("attributes", {})
            .get("price_usd")
        )
        if price is None:
            return None
        return Decimal(price)

    def token_balance_raw(self) -> int:
        return self.token.functions.balanceOf(self.wallet_address).call()

    def format_tokens(self, amount_raw: int) -> Decimal:
        return Decimal(amount_raw) / (Decimal(10) ** self.decimals)

    def should_sell(self, price_usd: Optional[Decimal]) -> Optional[str]:
        if price_usd is None:
            return None
        if self.cfg.take_profit_usd > 0 and price_usd >= self.cfg.take_profit_usd:
            return f"take-profit reached ({price_usd} >= {self.cfg.take_profit_usd})"
        if self.cfg.stop_loss_usd > 0 and price_usd <= self.cfg.stop_loss_usd:
            return f"stop-loss reached ({price_usd} <= {self.cfg.stop_loss_usd})"
        return None

    def _tx_params(self) -> dict:
        params = {
            "from": self.wallet_address,
            "nonce": self.web3.eth.get_transaction_count(self.wallet_address),
            "chainId": self.cfg.chain_id,
            "gas": self.cfg.gas_limit,
        }
        if self.cfg.gas_price_gwei > 0:
            params["gasPrice"] = self.web3.to_wei(self.cfg.gas_price_gwei, "gwei")
        return params

    def _sign_send_wait(self, tx: dict) -> str:
        signed = self.account.sign_transaction(tx)
        tx_hash = self.web3.eth.send_raw_transaction(signed.raw_transaction)
        tx_hex = tx_hash.hex()
        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
        if receipt.status != 1:
            raise RuntimeError(f"Transaction reverted: {tx_hex}")
        return tx_hex

    def ensure_approval(self, amount_raw: int) -> None:
        allowance = self.token.functions.allowance(
            self.wallet_address, self.router.address
        ).call()
        if allowance >= amount_raw:
            return

        if self.cfg.dry_run:
            print(f"[DRY RUN] Would approve router for {amount_raw} token units.")
            return

        max_uint = 2**256 - 1
        approve_tx = self.token.functions.approve(
            self.router.address,
            max_uint,
        ).build_transaction(self._tx_params())
        tx_hash = self._sign_send_wait(approve_tx)
        print(f"Approval sent: {tx_hash}")

    def quote_min_out(self, amount_in: int) -> int:
        quoted_out = self.router.functions.getAmountsOut(amount_in, self.path).call()[-1]
        min_out = quoted_out * (10_000 - self.cfg.slippage_bps) // 10_000
        return min_out

    def execute_sell(self, amount_in: int) -> str:
        amount_out_min = self.quote_min_out(amount_in)
        deadline = int(time.time()) + 180

        if self.cfg.use_fee_on_transfer:
            fn = self.router.functions.swapExactTokensForTokensSupportingFeeOnTransferTokens(
                amount_in, amount_out_min, self.path, self.wallet_address, deadline
            )
        else:
            fn = self.router.functions.swapExactTokensForTokens(
                amount_in, amount_out_min, self.path, self.wallet_address, deadline
            )

        tx = fn.build_transaction(self._tx_params())
        tx_hash = self._sign_send_wait(tx)
        return tx_hash

    def run_once(self, sell_anyway: bool = False) -> None:
        price_usd = self.get_token_price_usd()
        if price_usd is None:
            print("Could not fetch token USD price from GeckoTerminal.")
        else:
            print(f"Current token price: ${price_usd}")

        balance_raw = self.token_balance_raw()
        balance_human = self.format_tokens(balance_raw)
        print(f"Wallet token balance: {balance_human}")

        if balance_raw == 0:
            print("No token balance to sell.")
            return

        reason = "manual trigger" if sell_anyway else self.should_sell(price_usd)
        if reason is None:
            print("No sell condition met.")
            return

        amount_to_sell = int(Decimal(balance_raw) * self.cfg.sell_percent)
        if amount_to_sell <= 0:
            print("SELL_PERCENT produced 0 tokens to sell.")
            return

        print(f"Sell trigger: {reason}")
        print(f"Preparing to sell raw amount: {amount_to_sell}")
        self.ensure_approval(amount_to_sell)

        if self.cfg.dry_run:
            min_out = self.quote_min_out(amount_to_sell)
            print(
                "[DRY RUN] Would execute swap with amountIn="
                f"{amount_to_sell}, minOut={min_out}, path={self.path}"
            )
            return

        tx_hash = self.execute_sell(amount_to_sell)
        print(f"Sell tx sent and confirmed: {tx_hash}")

    def loop(self) -> None:
        while True:
            try:
                self.run_once()
                if self.cfg.sell_once:
                    break
            except Exception as exc:
                print(f"Error in loop: {exc}")
            time.sleep(self.cfg.check_interval_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rule-based crypto sell agent.")
    parser.add_argument(
        "--sell-now",
        action="store_true",
        help="Ignore price thresholds and sell immediately.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    cfg = AgentConfig.from_env()
    agent = SellAgent(cfg)

    if args.sell_now:
        agent.run_once(sell_anyway=True)
        return

    agent.loop()


if __name__ == "__main__":
    main()

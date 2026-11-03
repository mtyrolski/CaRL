"""Show shared campaign usage, committed reservations and local replay counts."""
import json

from carl.icl.campaign_budget import CampaignLedger


def main() -> int:
    """Print only aggregated ledger metadata, without requests or credentials."""
    print(json.dumps(CampaignLedger().summary(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

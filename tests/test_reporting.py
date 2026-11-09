import json

from fuel_elasticity.demo import make_demo_dataset
from fuel_elasticity.economics import OfferPolicy
from fuel_elasticity.reporting import save_results


def test_public_only_does_not_create_individual_data(tmp_path):
    ds = make_demo_dataset(120, seed=9)
    result = save_results(ds, tmp_path, OfferPolicy(), private_scores=False)
    assert result["dataset"]["clients"] == 120
    assert (tmp_path / "public" / "summary.json").is_file()
    assert (tmp_path / "public" / "summary.md").is_file()
    assert (tmp_path / "public" / "liters.svg").is_file()
    assert not (tmp_path / "private").exists()
    serialized = (tmp_path / "public" / "summary.json").read_text(encoding="utf8")
    assert "SYNTH_" not in serialized
    assert json.loads(serialized)["offer"]["eligible_clients"] >= 0


def test_private_is_separate_and_explicit(tmp_path):
    ds = make_demo_dataset(120, seed=9)
    save_results(ds, tmp_path, OfferPolicy(), private_scores=True)
    assert (tmp_path / "private" / "customer_scores.csv").exists()

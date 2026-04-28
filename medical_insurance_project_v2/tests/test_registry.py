"""Registry round-trip test."""
from sklearn.linear_model import LinearRegression
from src import registry

def test_register_and_load(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "REG_DIR", tmp_path)
    monkeypatch.setattr(registry, "ACTIVE_PTR", tmp_path / "active.json")
    model = LinearRegression()
    card = registry.register(model, name="dummy", algorithm="lr",
                             metrics={"r2":0.9}, params={"a":"1"},
                             feature_count=3, rows=10, dataset="x.csv")
    assert card.version == 1
    loaded = registry.load("dummy", 1)
    assert hasattr(loaded, "fit")

from grcshjepa.routing.graph import make_toy_routing_graph
from grcshjepa.routing.surface import routing_surface
from grcshjepa.routing.damage import apply_damage


def test_routing_surface_is_positive():
    graph = make_toy_routing_graph(20, 40, seed=0, variant="full_surface")
    metrics = routing_surface(graph)
    assert metrics["surface"] > 0
    assert metrics["delivered_traffic"] > 0


def test_load_targeted_damage_does_not_increase_traffic():
    graph = make_toy_routing_graph(20, 40, seed=0, variant="full_surface")
    base = routing_surface(graph)["delivered_traffic"]
    damaged = apply_damage(graph, "load_targeted", 0.10, seed=0)
    after = routing_surface(damaged)["delivered_traffic"]
    assert after <= base + 1e-8

import torch
import sys

# Add project src directories to Python path
sys.path.insert(0, "src")
sys.path.insert(0, "src/training")

from prepare_data import load_training_graph
from models.gcn_baseline import GCNBaseline


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("[INFO] Loading graph...")
    graph = load_training_graph()

    print(f"[INFO] Nodes: {graph.num_nodes}")
    print(f"[INFO] Features: {graph.x.shape}")
    print(f"[INFO] Edges: {graph.edge_index.shape}")

    # Same preprocessing used during training
    x = graph.x.clone()

    target_col = x[:, 0].clone()
    x[:, 0] = 0.0

    edge_index = graph.edge_index
    edge_weight = graph.edge_attr

    edge_weight = edge_weight / edge_weight.max()

    model = GCNBaseline(in_dim=x.shape[1]).to(DEVICE)

    model.load_state_dict(
        torch.load(
            "data/gcn_baseline_model.pt",
            map_location=DEVICE
        )
    )

    model.eval()

    x = x.to(DEVICE)
    edge_index = edge_index.to(DEVICE)
    edge_weight = edge_weight.to(DEVICE)

    with torch.no_grad():
        pred = model(x, edge_index, edge_weight)

    test_mask = graph.test_mask.to(DEVICE)

    print("[INFO] Model loaded successfully.")
    print(
        f"[INFO] Test predictions: "
        f"{test_mask.sum().item()}"
    )

    print(
        f"[INFO] Example prediction: "
        f"{pred[test_mask][0].item():.4f}"
    )

    print(
        f"[INFO] Example true SST: "
        f"{target_col[test_mask][0].item():.4f}"
    )


if __name__ == "__main__":
    main()
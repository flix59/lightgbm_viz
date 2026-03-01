"""LightGBM Tree Visualizer - Dash Application."""
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, Input, Output, State
import dash

from tree_extractor import TreeExtractor


# Load model and data
print("Loading model and data...")
with open('model.pkl', 'rb') as f:
    booster = pickle.load(f)

with open('train_data.pkl', 'rb') as f:
    X_train, y_train, X_test, y_test = pickle.load(f)

# Initialize tree extractor
extractor = TreeExtractor(booster, X_train, y_train)
num_trees = extractor.get_num_trees()

print(f"Loaded model with {num_trees} trees")

# Initialize Dash app
app = Dash(__name__)


def create_tree_plot(tree_idx: int) -> go.Figure:
    """Create Plotly figure for a specific tree."""
    tree = extractor.get_tree(tree_idx)
    node_samples = extractor.calculate_node_samples(tree_idx)

    # Build graph structure
    edge_x = []
    edge_y = []
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes = []

    # Layout tree using BFS with positioning
    positions = {}
    level_widths = {}

    def calculate_positions(node, depth=0, pos=0, parent_pos=None):
        """Calculate x, y positions for each node."""
        if 'split_index' not in node:
            # Leaf node
            node_id = f"leaf_{node.get('leaf_index', 'unknown')}"
            y = -depth
            x = pos

            positions[node_id] = (x, y)

            # Track level widths
            if depth not in level_widths:
                level_widths[depth] = []
            level_widths[depth].append(x)

            return 1  # width of this subtree

        # Split node
        node_id = f"split_{node['split_index']}"

        # Process children first to know their positions
        left_width = calculate_positions(node['left_child'], depth + 1, pos)
        right_width = calculate_positions(node['right_child'], depth + 1, pos + left_width)

        # Position this node centered above children
        left_pos = positions[f"split_{node['left_child'].get('split_index', 'unknown')}" if 'split_index' in node['left_child'] else f"leaf_{node['left_child'].get('leaf_index', 'unknown')}"]
        right_pos = positions[f"split_{node['right_child'].get('split_index', 'unknown')}" if 'split_index' in node['right_child'] else f"leaf_{node['right_child'].get('leaf_index', 'unknown')}"]

        x = (left_pos[0] + right_pos[0]) / 2
        y = -depth

        positions[node_id] = (x, y)

        if depth not in level_widths:
            level_widths[depth] = []
        level_widths[depth].append(x)

        return left_width + right_width

    # Calculate all positions
    calculate_positions(tree)

    # Build edges and nodes
    def add_node_and_edges(node, depth=0, parent_id=None):
        """Add node to plot and create edges."""
        if 'split_index' not in node:
            # Leaf node
            node_id = f"leaf_{node.get('leaf_index', 'unknown')}"
            sample_count = node_samples.get(node_id, 0)
            leaf_value = node.get('leaf_value', 0)

            x, y = positions[node_id]
            node_x.append(x)
            node_y.append(y)
            node_colors.append('lightgreen')
            node_sizes.append(max(20, min(50, sample_count / 10)))

            text = f"Leaf<br>Samples: {sample_count}<br>Value: {leaf_value:.2f}"
            node_text.append(text)

        else:
            # Split node
            node_id = f"split_{node['split_index']}"
            sample_count = node_samples.get(node_id, 0)
            feature_name = extractor.feature_names[node['split_feature']]
            threshold = node['threshold']
            gain = node.get('split_gain', 0)

            x, y = positions[node_id]
            node_x.append(x)
            node_y.append(y)
            node_colors.append('lightblue')
            node_sizes.append(max(20, min(50, sample_count / 10)))

            text = f"{feature_name}<br>Threshold: {threshold:.3f}<br>Samples: {sample_count}<br>Gain: {gain:.2f}"
            node_text.append(text)

            # Add edges to children and recurse
            for child_key in ['left_child', 'right_child']:
                child = node[child_key]
                if 'split_index' in child:
                    child_id = f"split_{child['split_index']}"
                else:
                    child_id = f"leaf_{child.get('leaf_index', 'unknown')}"

                child_pos = positions[child_id]

                # Add edge
                edge_x.extend([x, child_pos[0], None])
                edge_y.extend([y, child_pos[1], None])

                # Get child sample count for edge label
                child_samples = node_samples.get(child_id, 0)
                parent_samples = node_samples.get(node_id, 1)
                pct = (child_samples / parent_samples * 100) if parent_samples > 0 else 0

                # Recurse
                add_node_and_edges(child, depth + 1, node_id)

    # Build the graph
    add_node_and_edges(tree)

    # Create figure
    fig = go.Figure()

    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=2, color='gray'),
        hoverinfo='none',
        showlegend=False
    ))

    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='darkgray')
        ),
        text=node_text,
        hoverinfo='text',
        textposition='top center',
        showlegend=False
    ))

    # Update layout
    summary = extractor.get_tree_summary(tree_idx)
    fig.update_layout(
        title=f"Tree {tree_idx} | Nodes: {summary['total_nodes']} (Splits: {summary['split_nodes']}, Leaves: {summary['leaf_nodes']})",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=80),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=700,
        plot_bgcolor='white'
    )

    return fig


def create_feature_importance_plot() -> go.Figure:
    """Create feature importance bar chart."""
    importance_df = extractor.get_feature_importance()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=importance_df['importance'],
        y=importance_df['feature'],
        orientation='h',
        marker=dict(color='steelblue')
    ))

    fig.update_layout(
        title="Feature Importance (Gain)",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=400,
        margin=dict(l=200, r=20, t=50, b=50)
    )

    return fig


# App layout
app.layout = html.Div([
    html.H1("🌳 LightGBM Tree Visualizer",
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),

    html.Div([
        html.Div([
            html.H3(f"Model: {num_trees} Trees", style={'color': '#34495e'}),
            html.P(f"Features: {len(extractor.feature_names)}", style={'color': '#7f8c8d'}),
            html.P(f"Training Samples: {len(X_train)}", style={'color': '#7f8c8d'}),
        ], style={'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '5px',
                  'marginBottom': '20px'}),

        # Feature importance
        html.Div([
            dcc.Graph(id='feature-importance', figure=create_feature_importance_plot())
        ], style={'marginBottom': '30px'}),

        # Tree navigation
        html.Div([
            html.Label("Select Tree:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
            dcc.Slider(
                id='tree-slider',
                min=0,
                max=num_trees - 1,
                step=1,
                value=0,
                marks={i: str(i) for i in range(0, num_trees, max(1, num_trees // 10))},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={'marginBottom': '30px', 'padding': '20px',
                  'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),

        # Tree visualization
        html.Div([
            dcc.Graph(id='tree-plot')
        ]),

        # Tree statistics
        html.Div(id='tree-stats',
                style={'marginTop': '20px', 'padding': '20px',
                       'backgroundColor': '#e8f5e9', 'borderRadius': '5px'})

    ], style={'maxWidth': '1400px', 'margin': '0 auto', 'padding': '20px'})
])


@callback(
    Output('tree-plot', 'figure'),
    Output('tree-stats', 'children'),
    Input('tree-slider', 'value')
)
def update_tree(tree_idx):
    """Update tree visualization when slider changes."""
    fig = create_tree_plot(tree_idx)
    summary = extractor.get_tree_summary(tree_idx)

    stats_div = html.Div([
        html.H4(f"Tree {tree_idx} Statistics", style={'color': '#2e7d32'}),
        html.P(f"Total Nodes: {summary['total_nodes']}"),
        html.P(f"Split Nodes: {summary['split_nodes']}"),
        html.P(f"Leaf Nodes: {summary['leaf_nodes']}"),
        html.P(f"Samples Processed: {summary['total_samples']}")
    ])

    return fig, stats_div


if __name__ == '__main__':
    print(f"\n{'='*60}")
    print("Starting LightGBM Tree Visualizer")
    print(f"{'='*60}")
    print(f"Trees available: {num_trees}")
    print(f"Features: {extractor.feature_names}")
    print(f"\nNavigate to http://127.0.0.1:8050 in your browser")
    print(f"{'='*60}\n")

    app.run(debug=True)

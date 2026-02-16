"""
Character Relationship Visualizer
Creates interactive network graphs using Plotly.
"""

import plotly.graph_objects as go
import networkx as nx
from typing import Dict, List, Any


def create_relationship_graph(graph_data: Dict[str, Any]) -> go.Figure:
    """
    Create interactive Plotly graph from character relationship data.
    
    Args:
        graph_data: {
            'nodes': [{id, label, ...}],
            'edges': [{from, to, label, color, ...}]
        }
        
    Returns:
        Plotly Figure object
    """
    nodes = graph_data.get('nodes', [])
    edges = graph_data.get('edges', [])
    
    if not nodes:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No characters detected in the story",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        return fig
    
    # Build NetworkX graph for layout
    G = nx.Graph()
    
    # Add nodes
    for node in nodes:
        G.add_node(node['id'])
    
    # Add edges
    for edge in edges:
        G.add_edge(edge['from'], edge['to'])
    
    # Compute layout (spring layout for nice spacing)
    try:
        pos = nx.spring_layout(G, k=2, iterations=50)
    except:
        # Fallback if graph is too small
        pos = {node['id']: (i, 0) for i, node in enumerate(nodes)}
    
    # Build Plotly traces
    edge_traces = []
    edge_annotations = []
    
    # Create edges with offset labels to prevent overlap
    for idx, edge in enumerate(edges):
        x0, y0 = pos[edge['from']]
        x1, y1 = pos[edge['to']]
        
        # Edge line
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=2, color=edge['color']),
            hoverinfo='text',
            hovertext=f"{edge['from']} ↔ {edge['to']}<br>{edge['label']}<br>{edge['description']}",
            showlegend=False
        )
        edge_traces.append(edge_trace)
        
        # Calculate label position with offset to prevent overlap
        # Base position: slightly offset from midpoint
        t = 0.5 + (idx % 3 - 1) * 0.1  # Vary position along edge: 0.4, 0.5, 0.6
        label_x = x0 + t * (x1 - x0)
        label_y = y0 + t * (y1 - y0)
        
        # Add perpendicular offset based on edge index
        dx = x1 - x0
        dy = y1 - y0
        length = (dx**2 + dy**2)**0.5
        if length > 0:
            # Perpendicular unit vector
            perp_x = -dy / length
            perp_y = dx / length
            # Offset amount varies by index
            offset = (idx % 5 - 2) * 0.08  # Offsets: -0.16, -0.08, 0, 0.08, 0.16
            label_x += perp_x * offset
            label_y += perp_y * offset
        
        edge_annotations.append(
            dict(
                x=label_x, y=label_y,
                text=edge['label'],
                showarrow=False,
                font=dict(size=8, color=edge['color']),
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor=edge['color'],
                borderwidth=1,
                borderpad=2,
                opacity=0.95
            )
        )
    
    # Create nodes
    node_x = []
    node_y = []
    node_text = []
    node_hover = []
    
    for node in nodes:
        x, y = pos[node['id']]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node['label'])
        
        # Count relationships for this character
        rel_count = sum(1 for e in edges if e['from'] == node['id'] or e['to'] == node['id'])
        node_hover.append(f"{node['label']}<br>{rel_count} relationship(s)")
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        hovertext=node_hover,
        marker=dict(
            size=25,
            color='#3498db',  # Blue
            line=dict(width=2, color='white')
        ),
        textfont=dict(size=12, color='black', family='Arial Black'),
        showlegend=False
    )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    
    # Add edge labels
    fig.update_layout(annotations=edge_annotations)
    
    # Layout settings
    fig.update_layout(
        title=dict(
            text="Character Relationship Map",
            x=0.5,
            xanchor='center',
            font=dict(size=20, color='#2c3e50')
        ),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=20, r=20, t=60),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='#f8f9fa',
        height=600
    )
    
    return fig


def create_legend_html() -> str:
    """Create HTML legend for relationship types."""
    
    legend_items = [
        ("🩸 Blood Relations", "#FF6B6B", "sibling, parent, child, cousin, etc."),
        ("💕 Romance", "#FF69B4", "romantic, lovers, ex-lovers"),
        ("🤝 Friendship", "#4ECDC4", "friend, best friend, ally"),
        ("👨‍👩‍👧 Family (non-blood)", "#FFA500", "adopted, step-sibling, in-law"),
        ("⚔️ Antagonistic", "#8B0000", "enemy, rival"),
        ("💼 Professional", "#9B59B6", "mentor, student, colleague, boss"),
    ]
    
    html = "<div style='background: white; padding: 15px; border-radius: 8px; margin-top: 10px;'>"
    html += "<h3 style='margin-top: 0; color: #2c3e50;'>Relationship Types</h3>"
    html += "<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px;'>"
    
    for label, color, desc in legend_items:
        html += f"""
        <div style='display: flex; align-items: center; padding: 5px;'>
            <div style='width: 20px; height: 20px; background: {color}; 
                        border-radius: 3px; margin-right: 10px;'></div>
            <div>
                <strong>{label}</strong><br>
                <small style='color: #7f8c8d;'>{desc}</small>
            </div>
        </div>
        """
    
    html += "</div></div>"
    return html

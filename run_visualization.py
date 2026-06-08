import os
import sys
import json
import networkx as nx

# Add current directory to path to import src modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.graph_builder import GraphBuilder

def extract_insights(G: nx.MultiDiGraph):
    print("\n====================================================")
    print("                STRUCTURAL INSIGHTS                 ")
    print("====================================================")
    
    # 1. Top Central Tools (Degree Centrality)
    degree = dict(G.degree())
    sorted_degree = sorted(degree.items(), key=lambda x: x[1], reverse=True)
    
    print("\n--- Top 10 Most Connected Tools (Degree Centrality) ---")
    for i, (node, deg) in enumerate(sorted_degree[:10]):
        print(f"{i+1}. {node} (Edges: {deg})")
        
    # 2. Compositional Dependency Chains (Longest prerequisite chains)
    dep_edges = [(u, v) for u, v, k, d in G.edges(keys=True, data=True) if k == 'compose_dep']
    dep_graph = nx.DiGraph(dep_edges)
    
    print(f"\nTotal Tools in Dependency Chains: {dep_graph.number_of_nodes()}")
    print(f"Total Dependency Links Mined: {dep_graph.number_of_edges()}")
    
    # Find longest paths in dependency graph if it is a DAG
    if nx.is_directed_acyclic_graph(dep_graph):
        try:
            longest_path = nx.dag_longest_path(dep_graph)
            print("\n--- Longest Prerequisite Chain Found ---")
            print(" -> ".join(longest_path))
        except Exception as e:
            print(f"\nCould not calculate longest DAG path: {e}")
    else:
        try:
            cycles = list(nx.simple_cycles(dep_graph))
            print(f"\nFound {len(cycles)} dependency circular loops (cycles). Top 3:")
            for c in cycles[:3]:
                print(" -> ".join(c) + " -> " + c[0])
        except Exception as e:
            print(f"\nCould not check for cycles: {e}")
            
    # 3. Server-Level Aggregation
    server_tools = {}
    for node, data in G.nodes(data=True):
        server = data.get("server", "unknown")
        server_tools[server] = server_tools.get(server, 0) + 1
        
    sorted_servers = sorted(server_tools.items(), key=lambda x: x[1], reverse=True)
    print("\n--- Top 5 Largest Servers by Tool Count ---")
    for i, (server, count) in enumerate(sorted_servers[:5]):
         print(f"{i+1}. {server}: {count} tools")

def generate_interactive_html(G: nx.MultiDiGraph, output_path: str):
    vis_nodes = []
    vis_edges = []
    
    import random
    random.seed(42)
    server_colors = {}
    
    def get_color(server):
        if server not in server_colors:
            h = random.randint(0, 360)
            s = random.randint(75, 95)
            l = random.randint(45, 60)
            server_colors[server] = f"hsl({h}, {s}%, {l}%)"
        return server_colors[server]
        
    # Find nodes that have at least one compositional dependency edge
    dep_nodes = set()
    for u, v, k, d in G.edges(keys=True, data=True):
        if k == 'compose_dep':
            dep_nodes.add(u)
            dep_nodes.add(v)
            
    if not dep_nodes:
        degree = dict(G.degree())
        sorted_nodes = sorted(degree.items(), key=lambda x: x[1], reverse=True)
        dep_nodes = {n[0] for n in sorted_nodes[:100]}
        
    # Collect nodes
    for node in dep_nodes:
        data = G.nodes[node]
        server = data.get("server", "unknown")
        desc = data.get("description", "")
        params = data.get("parameter_names", [])
        
        title = f"<b>Tool:</b> {data.get('name')}<br><b>Server:</b> {server}<br><b>Params:</b> {', '.join(params) if params else 'None'}<br><br>{desc}"
        
        vis_nodes.append({
            "id": node,
            "label": data.get("name"),
            "group": server,
            "color": get_color(server),
            "title": title,
            "description": desc,
            "params": params,
            "value": len(params) + 5
        })
        
    # Collect edges
    edge_seen = set()
    for u, v, k, d in G.edges(keys=True, data=True):
        if u in dep_nodes and v in dep_nodes:
            edge_id = f"{u}->{v}:{k}"
            if edge_id in edge_seen:
                continue
            edge_seen.add(edge_id)
            
            # We only render composition dependency edges to keep visualization fast and clean.
            if k == 'compose_dep':
                vis_edges.append({
                    "from": u,
                    "to": v,
                    "arrows": "to",
                    "color": {"color": "#ff4757", "highlight": "#ff6b81"},
                    "width": 3,
                    "label": "prereq",
                    "font": {"size": 8, "color": "#ff4757"},
                    "title": f"Dependency Confidence: {d['weight']:.2f}",
                    "weight": d['weight']
                })

    # Pre-calculate structural insights for dashboard display
    degree = dict(G.degree())
    sorted_degree = sorted(degree.items(), key=lambda x: x[1], reverse=True)
    top_connected_python = [
        {"id": n, "name": G.nodes[n].get("name", n), "server": G.nodes[n].get("server", "unknown"), "edges": d} 
        for n, d in sorted_degree[:10] if n in dep_nodes
    ]
    
    server_tools = {}
    for node, data in G.nodes(data=True):
        server = data.get("server", "unknown")
        server_tools[server] = server_tools.get(server, 0) + 1
    sorted_servers = sorted(server_tools.items(), key=lambda x: x[1], reverse=True)
    top_servers_python = [{"server": s, "count": c} for s, c in sorted_servers[:10]]
    
    dep_edges = [(u, v) for u, v, k, d in G.edges(keys=True, data=True) if k == 'compose_dep']
    dep_graph = nx.DiGraph(dep_edges)
    
    cycles_python = []
    if not nx.is_directed_acyclic_graph(dep_graph):
        try:
            cycles = list(nx.simple_cycles(dep_graph))
            cycles = sorted(cycles, key=len)
            for cycle in cycles[:5]:
                cycles_python.append([
                    {"id": node, "name": G.nodes[node].get("name", node), "server": G.nodes[node].get("server", "unknown")} 
                    for node in cycle
                ])
        except Exception:
            pass

    longest_chain_python = []
    if nx.is_directed_acyclic_graph(dep_graph):
        try:
            chain = nx.dag_longest_path(dep_graph)
            longest_chain_python = [
                {"id": node, "name": G.nodes[node].get("name", node), "server": G.nodes[node].get("server", "unknown")} 
                for node in chain
            ]
        except Exception:
            pass

    insights = {
        "top_connected": top_connected_python,
        "top_servers": top_servers_python,
        "cycles": cycles_python,
        "longest_chain": longest_chain_python,
        "total_all_nodes": G.number_of_nodes(),
        "total_all_edges": G.number_of_edges(),
        "vis_nodes_count": len(vis_nodes),
        "vis_edges_count": len(vis_edges)
    }

    # Generate Legend HTML
    legend_html_list = []
    for server, color in list(server_colors.items())[:12]:
        legend_html_list.append(
            f'<div class="legend-item"><div class="legend-color" style="background-color: {color}"></div><span>{server}</span></div>'
        )
    if len(server_colors) > 12:
        legend_html_list.append(
            f'<div class="legend-item" style="font-style: italic; color: #a4b0be;"><span>... and {len(server_colors)-12} more servers</span></div>'
        )
    legend_html = "".join(legend_html_list)

    # HTML dashboard template
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FED-GRAPH-MCP Tool Dependency Dashboard</title>
    <!-- vis-network jsDelivr CDN -->
    <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/vis-network@9.1.2/dist/vis-network.min.js"></script>
    <!-- Outfit & Inter Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <!-- FontAwesome Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <style>
        :root {
            --bg-main: #0c0e12;
            --bg-card: rgba(20, 24, 33, 0.75);
            --bg-card-hover: rgba(28, 34, 46, 0.9);
            --border-glow: rgba(69, 243, 255, 0.15);
            --border-glow-active: rgba(69, 243, 255, 0.4);
            --color-text-primary: #f1f2f6;
            --color-text-secondary: #a4b0be;
            --accent-cyan: #45f3ff;
            --accent-coral: #ff4757;
            --accent-emerald: #2ed573;
            --sidebar-width: 400px;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-main);
            color: var(--color-text-primary);
            margin: 0;
            padding: 0;
            overflow: hidden;
            display: flex;
            height: 100vh;
            width: 100vw;
        }

        /* Loading Screen */
        #loading-screen {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(12, 14, 18, 0.95);
            z-index: 9999;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            transition: opacity 0.5s ease-out;
        }
        .spinner {
            width: 70px;
            height: 70px;
            border: 4px solid transparent;
            border-top: 4px solid var(--accent-cyan);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            position: relative;
            margin-bottom: 20px;
        }
        .spinner::after {
            content: '';
            position: absolute;
            top: 5px; left: 5px; right: 5px; bottom: 5px;
            border: 4px solid transparent;
            border-top: 4px solid var(--accent-coral);
            border-radius: 50%;
            animation: spin 2s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .loading-title {
            font-family: 'Outfit', sans-serif;
            font-size: 24px;
            font-weight: 600;
            color: var(--accent-cyan);
            text-shadow: 0 0 10px rgba(69, 243, 255, 0.3);
            margin-bottom: 8px;
        }
        .loading-subtitle {
            font-size: 14px;
            color: var(--color-text-secondary);
        }
        .loading-progress {
            margin-top: 15px;
            font-size: 12px;
            color: var(--accent-coral);
        }

        /* Main Workspace Layout */
        #workspace {
            display: flex;
            width: 100vw;
            height: 100vh;
            position: relative;
            flex-grow: 1;
        }

        #network-container {
            flex-grow: 1;
            height: 100%;
            position: relative;
            background-color: #08090c;
        }

        #network {
            width: 100%;
            height: 100%;
        }

        /* Header Control Overlay */
        .header-panel {
            position: absolute;
            top: 20px;
            left: 20px;
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-glow);
            padding: 15px 25px;
            border-radius: 12px;
            z-index: 10;
            pointer-events: auto;
            max-width: 480px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
            transition: border-color 0.3s;
        }
        .header-panel:hover {
            border-color: var(--border-glow-active);
        }
        .header-panel h1 {
            margin: 0;
            font-family: 'Outfit', sans-serif;
            font-size: 22px;
            font-weight: 800;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, #fff, var(--accent-cyan));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .header-panel p {
            margin: 5px 0 12px 0;
            font-size: 13px;
            color: var(--color-text-secondary);
            line-height: 1.4;
        }
        .stats-badge-row {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        .stats-badge {
            background: rgba(255, 255, 255, 0.05);
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        .stats-badge .accent {
            color: var(--accent-cyan);
        }

        /* Floating Toolbar (Bottom Left) */
        .toolbar {
            position: absolute;
            bottom: 25px;
            left: 25px;
            display: flex;
            align-items: center;
            gap: 12px;
            z-index: 10;
        }
        .search-box {
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-glow);
            border-radius: 10px;
            padding: 5px 12px;
            display: flex;
            align-items: center;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
            width: 280px;
        }
        .search-box input {
            background: transparent;
            border: none;
            color: var(--color-text-primary);
            padding: 8px;
            outline: none;
            width: 100%;
            font-size: 13px;
        }
        .search-box button {
            background: transparent;
            border: none;
            color: var(--accent-cyan);
            cursor: pointer;
            padding: 8px;
            font-size: 15px;
            transition: transform 0.2s;
        }
        .search-box button:hover {
            transform: scale(1.15);
        }
        .control-btn {
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-glow);
            color: var(--color-text-primary);
            width: 44px;
            height: 44px;
            border-radius: 10px;
            cursor: pointer;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 16px;
            transition: all 0.3s;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        }
        .control-btn:hover {
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
            transform: translateY(-2px);
        }
        .control-btn.active {
            background: var(--accent-cyan);
            color: var(--bg-main);
            border-color: var(--accent-cyan);
            box-shadow: 0 0 15px rgba(69, 243, 255, 0.4);
        }

        /* Right Panel (Sidebar) */
        #sidebar {
            width: var(--sidebar-width);
            height: 100vh;
            background: rgba(16, 20, 28, 0.85);
            backdrop-filter: blur(20px);
            border-left: 1px solid rgba(255, 255, 255, 0.08);
            display: flex;
            flex-direction: column;
            z-index: 100;
            box-shadow: -10px 0 30px rgba(0, 0, 0, 0.5);
        }

        /* Sidebar Tabs */
        .sidebar-tabs {
            display: flex;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(0, 0, 0, 0.2);
        }
        .tab-btn {
            flex: 1;
            padding: 18px;
            background: transparent;
            border: none;
            color: var(--color-text-secondary);
            font-family: 'Outfit', sans-serif;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 8px;
        }
        .tab-btn:hover {
            color: var(--color-text-primary);
            background: rgba(255, 255, 255, 0.02);
        }
        .tab-btn.active {
            color: var(--accent-cyan);
            border-bottom: 2px solid var(--accent-cyan);
            background: rgba(69, 243, 255, 0.03);
            text-shadow: 0 0 8px rgba(69, 243, 255, 0.2);
        }

        /* Tab Contents */
        .tab-content-container {
            flex-grow: 1;
            overflow-y: auto;
            padding: 25px;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
            animation: fadeIn 0.4s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(5deg); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Legend floating over graph */
        .legend-card {
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-glow);
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
            margin-bottom: 20px;
        }
        .legend-title {
            font-family: 'Outfit', sans-serif;
            font-size: 13px;
            font-weight: 700;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--accent-cyan);
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            padding-bottom: 5px;
        }
        .legend-list {
            max-height: 200px;
            overflow-y: auto;
            padding-right: 5px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            font-size: 12px;
        }
        .legend-color {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 10px;
            box-shadow: 0 0 5px rgba(255,255,255,0.2);
        }

        /* Inspector Details Elements */
        .detail-header {
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            padding-bottom: 15px;
            margin-bottom: 20px;
        }
        .detail-header h2 {
            margin: 0;
            font-family: 'Outfit', sans-serif;
            font-size: 24px;
            font-weight: 700;
            color: #fff;
            word-break: break-all;
        }
        .detail-server-tag {
            display: inline-block;
            background: rgba(69, 243, 255, 0.1);
            color: var(--accent-cyan);
            border: 1px solid rgba(69, 243, 255, 0.2);
            padding: 4px 10px;
            border-radius: 100px;
            font-size: 11px;
            font-weight: 600;
            margin-top: 8px;
        }
        .detail-section {
            margin-bottom: 20px;
        }
        .detail-section-title {
            font-family: 'Outfit', sans-serif;
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
            color: var(--color-text-secondary);
            margin-bottom: 8px;
            letter-spacing: 0.5px;
        }
        .detail-body {
            font-size: 13.5px;
            line-height: 1.6;
            color: var(--color-text-primary);
            background: rgba(255, 255, 255, 0.02);
            padding: 12px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.04);
            white-space: pre-wrap;
        }
        .param-list {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }
        .param-tag {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 5px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-family: monospace;
            color: #ff9f43;
        }
        .relation-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .relation-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12.5px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .relation-item:hover {
            background: rgba(69, 243, 255, 0.08);
            border-color: rgba(69, 243, 255, 0.3);
            transform: translateX(3px);
        }
        .relation-item span {
            font-weight: 500;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            max-width: 250px;
        }
        .relation-item .arrow-badge {
            font-size: 10px;
            background: rgba(255, 71, 87, 0.15);
            color: var(--accent-coral);
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: 600;
        }

        /* Insight Tab Elements */
        .insight-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .insight-card h3 {
            margin: 0 0 12px 0;
            font-family: 'Outfit', sans-serif;
            font-size: 15px;
            font-weight: 700;
            color: var(--accent-cyan);
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding-bottom: 8px;
        }
        .insight-row {
            display: flex;
            justify-content: space-between;
            font-size: 12.5px;
            margin-bottom: 8px;
            border-bottom: 1px dashed rgba(255, 255, 255, 0.03);
            padding-bottom: 6px;
        }
        .insight-row:last-child {
            margin-bottom: 0;
            border-bottom: none;
            padding-bottom: 0;
        }
        .insight-row span:first-child {
            color: var(--color-text-secondary);
        }
        .insight-row span:last-child {
            font-weight: 600;
        }
        .interactive-chain-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .chain-step {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 12.5px;
            background: rgba(255, 255, 255, 0.02);
            padding: 8px 12px;
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.04);
            cursor: pointer;
            transition: all 0.2s;
        }
        .chain-step:hover {
            border-color: var(--accent-cyan);
            background: rgba(69, 243, 255, 0.05);
        }
        .chain-step .number {
            background: var(--accent-cyan);
            color: var(--bg-main);
            width: 20px;
            height: 20px;
            border-radius: 50%;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 10px;
            font-weight: 800;
        }

        /* Scrollbar configuration */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.1);
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.15);
            border-radius: 100px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(69, 243, 255, 0.4);
        }

        .default-msg {
            color: var(--color-text-secondary);
            text-align: center;
            font-style: italic;
            padding: 40px 10px;
            font-size: 13.5px;
        }
    </style>
</head>
<body>

    <!-- Loading Modal -->
    <div id="loading-screen">
        <div class="spinner"></div>
        <div class="loading-title">Analyzing MCP Topology</div>
        <div class="loading-subtitle">Stabilizing tool dependency network...</div>
        <div id="progress-text" class="loading-progress">Initializing physics layout...</div>
    </div>

    <div id="workspace">
        <div id="network-container">
            <!-- Floating Header Panel -->
            <div class="header-panel">
                <h1>FED-GRAPH-MCP Visualizer</h1>
                <p>An interactive visualization of compositional tool dependencies. Red arrows represent required prerequisite chains mined from trajectories.</p>
                <div class="stats-badge-row">
                    <div class="stats-badge">Nodes: <span class="accent">__TOTAL_VIS_NODES__</span></div>
                    <div class="stats-badge">Edges: <span class="accent">__TOTAL_VIS_EDGES__</span></div>
                    <div class="stats-badge" style="border-color: rgba(46, 213, 115, 0.2);">Heuristic Provider</div>
                </div>
            </div>

            <!-- Floating Controls toolbar -->
            <div class="toolbar">
                <div class="search-box">
                    <input type="text" id="node-search" placeholder="Search tool or server..." onkeydown="if(event.key === 'Enter') executeSearch()">
                    <button onclick="executeSearch()"><i class="fa-solid fa-magnifying-glass"></i></button>
                </div>
                <button id="physics-btn" class="control-btn active" onclick="togglePhysics(this)" title="Toggle Physics Simulation">
                    <i class="fa-solid fa-play"></i>
                </button>
                <button class="control-btn" onclick="network.fit({animation:true})" title="Fit Graph to Screen">
                    <i class="fa-solid fa-compress"></i>
                </button>
            </div>

            <div id="network"></div>
        </div>

        <!-- Right Side Panel -->
        <div id="sidebar">
            <div class="sidebar-tabs">
                <button class="tab-btn active" onclick="switchTab('inspector', this)">
                    <i class="fa-solid fa-circle-info"></i> Tool Inspector
                </button>
                <button class="tab-btn" onclick="switchTab('insights', this)">
                    <i class="fa-solid fa-chart-pie"></i> Insights
                </button>
            </div>

            <div class="tab-content-container">
                <!-- Inspector Content -->
                <div id="tab-inspector" class="tab-content active">
                    <div id="inspector-placeholder" class="default-msg">
                        <i class="fa-solid fa-arrow-pointer" style="font-size: 32px; margin-bottom: 12px; color: var(--accent-cyan); opacity:0.8;"></i>
                        <br>Select a tool node on the graph to inspect parameters, descriptions, and functional dependencies.
                    </div>
                    <div id="inspector-content" style="display: none;">
                        <div class="detail-header">
                            <h2 id="inspect-name">create_table</h2>
                            <span id="inspect-server" class="detail-server-tag">Sqlite</span>
                        </div>
                        <div class="detail-section">
                            <div class="detail-section-title">Functional Description</div>
                            <div id="inspect-desc" class="detail-body">Create new tables in the database.</div>
                        </div>
                        <div class="detail-section">
                            <div class="detail-section-title">Required Parameter Schema</div>
                            <div id="inspect-params" class="param-list">
                                <span class="param-tag">query</span>
                            </div>
                        </div>
                        <div class="detail-section">
                            <div class="detail-section-title">Prerequisites (Required Before)</div>
                            <div id="inspect-prereqs" class="relation-list">
                                <!-- Dynamic relation items -->
                            </div>
                        </div>
                        <div class="detail-section">
                            <div class="detail-section-title">Dependents (Required After)</div>
                            <div id="inspect-dependents" class="relation-list">
                                <!-- Dynamic relation items -->
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Insights Content -->
                <div id="tab-insights" class="tab-content">
                    <!-- Legend inside sidebar -->
                    <div class="legend-card">
                        <div class="legend-title">Servers List</div>
                        <div class="legend-list">
                            __LEGEND_ITEMS__
                        </div>
                    </div>

                    <!-- Density metrics -->
                    <div class="insight-card">
                        <h3>Global Metrics</h3>
                        <div class="insight-row">
                            <span>Total Tools Mined:</span>
                            <span id="stat-total-nodes">0</span>
                        </div>
                        <div class="insight-row">
                            <span>Total Functional Edges:</span>
                            <span id="stat-total-edges">0</span>
                        </div>
                        <div class="insight-row">
                            <span>Visualization Nodes:</span>
                            <span>__TOTAL_VIS_NODES__</span>
                        </div>
                        <div class="insight-row">
                            <span>Visualization Edges:</span>
                            <span>__TOTAL_VIS_EDGES__</span>
                        </div>
                    </div>

                    <!-- Top Servers -->
                    <div class="insight-card">
                        <h3>Top 5 Largest Servers</h3>
                        <div id="top-servers-list">
                            <!-- Populated dynamically -->
                        </div>
                    </div>

                    <!-- Longest Chain -->
                    <div class="insight-card">
                        <h3>Longest Dependency Chain</h3>
                        <div id="longest-chain-section" class="interactive-chain-list">
                            <!-- Populated dynamically -->
                        </div>
                    </div>

                    <!-- Circular dependency loops -->
                    <div class="insight-card" id="cycles-card" style="display:none;">
                        <h3>Detected Circular Loops</h3>
                        <div id="cycles-section" class="interactive-chain-list">
                            <!-- Populated dynamically -->
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Data Injection -->
    <script type="text/javascript">
        var nodesData = __NODES_DATA__;
        var edgesData = __EDGES_DATA__;
        var insightsData = __INSIGHTS_DATA__;
    </script>

    <!-- App Logic -->
    <script type="text/javascript">
        // Safe check for vis library loading
        if (typeof vis === 'undefined') {
            document.body.innerHTML = '<div style="color:#ff4757; text-align:center; padding:50px; font-family:sans-serif;">' +
                                      '<h2>vis.js CDN Library failed to load!</h2>' +
                                      '<p>Please check your internet connection or try loading this page in a browser with access to cdn.jsdelivr.net</p>' +
                                      '</div>';
        }

        // Initialize datasets
        var nodes = new vis.DataSet(nodesData);
        var edges = new vis.DataSet(edgesData);
        var container = document.getElementById('network');
        var data = { nodes: nodes, edges: edges };

        // Configuration options
        var options = {
            nodes: {
                shape: 'dot',
                scaling: {
                    min: 10,
                    max: 30
                },
                font: {
                    size: 11,
                    face: 'Inter, sans-serif',
                    color: '#c2c4db'
                },
                borderWidth: 2,
                shadow: {
                    enabled: true,
                    color: 'rgba(0,0,0,0.5)',
                    size: 8,
                    x: 3,
                    y: 3
                }
            },
            edges: {
                smooth: {
                    type: 'continuous',
                    roundness: 0.4
                },
                shadow: true,
                selectionWidth: 2
            },
            physics: {
                barnesHut: {
                    gravitationalConstant: -1800,
                    centralGravity: 0.15,
                    springLength: 90,
                    springConstant: 0.05,
                    damping: 0.09,
                    avoidOverlap: 0.15
                },
                maxVelocity: 45,
                minVelocity: 0.1,
                solver: 'barnesHut',
                stabilization: {
                    enabled: true,
                    iterations: 150,
                    updateInterval: 25
                }
            }
        };

        // Create Network
        var network = new vis.Network(container, data, options);
        var isPhysicsPlaying = true;

        // Stabilisation loading screen handler
        network.on("stabilizationProgress", function(params) {
            var progress = Math.round((params.iterations / params.total) * 100);
            document.getElementById('progress-text').innerText = 'Stabilizing layout: ' + progress + '% (' + params.iterations + '/' + params.total + ' iterations)';
        });

        network.on("stabilizationFinished", function() {
            var loader = document.getElementById('loading-screen');
            loader.style.opacity = 0;
            setTimeout(function() {
                loader.style.display = 'none';
            }, 500);
        });

        // Fallback for loader if stabilization hangs
        setTimeout(function() {
            var loader = document.getElementById('loading-screen');
            if (loader.style.display !== 'none') {
                loader.style.opacity = 0;
                setTimeout(function() { loader.style.display = 'none'; }, 500);
            }
        }, 3000);

        // Sidebar Navigation Tabs
        function switchTab(tabName, el) {
            // Remove active classes
            var tabs = document.getElementsByClassName('tab-btn');
            for(var i=0; i<tabs.length; i++) {
                tabs[i].classList.remove('active');
            }
            var contents = document.getElementsByClassName('tab-content');
            for(var i=0; i<contents.length; i++) {
                contents[i].classList.remove('active');
            }
            
            // Add active class
            el.classList.add('active');
            document.getElementById('tab-' + tabName).classList.add('active');
        }

        // Toggle Physics Play/Pause
        function togglePhysics(btn) {
            isPhysicsPlaying = !isPhysicsPlaying;
            network.setOptions({ physics: { enabled: isPhysicsPlaying } });
            
            if (isPhysicsPlaying) {
                btn.classList.add('active');
                btn.innerHTML = '<i class="fa-solid fa-play"></i>';
            } else {
                btn.classList.remove('active');
                btn.innerHTML = '<i class="fa-solid fa-pause"></i>';
            }
        }

        // Search Tool Handler
        function executeSearch() {
            var searchVal = document.getElementById('node-search').value.toLowerCase().trim();
            if (!searchVal) return;

            var matchId = null;
            nodes.forEach(function(node) {
                if (node.label.toLowerCase().includes(searchVal) || 
                    node.id.toLowerCase().includes(searchVal) ||
                    (node.group && node.group.toLowerCase().includes(searchVal))) {
                    matchId = node.id;
                }
            });

            if (matchId) {
                network.selectNodes([matchId]);
                network.focus(matchId, {
                    scale: 1.3,
                    animation: {
                        duration: 1000,
                        easingFunction: "easeInOutQuad"
                    }
                });
                showNodeDetails(matchId);
                switchTab('inspector', document.querySelector('.tab-btn:first-child'));
            } else {
                alert("No tool or server matches search: " + searchVal);
            }
        }

        // Click focus helper
        function focusOnNode(nodeId) {
            network.selectNodes([nodeId]);
            network.focus(nodeId, {
                scale: 1.2,
                animation: {
                    duration: 800,
                    easingFunction: "easeInOutQuad"
                }
            });
            showNodeDetails(nodeId);
            switchTab('inspector', document.querySelector('.tab-btn:first-child'));
        }

        // Inspector Populating Logic
        network.on("click", function(params) {
            if (params.nodes.length > 0) {
                var nodeId = params.nodes[0];
                showNodeDetails(nodeId);
            } else {
                clearInspector();
            }
        });

        function showNodeDetails(nodeId) {
            var node = nodes.get(nodeId);
            if (!node) return;

            document.getElementById('inspector-placeholder').style.display = 'none';
            document.getElementById('inspector-content').style.display = 'block';

            document.getElementById('inspect-name').innerText = node.label;
            document.getElementById('inspect-server').innerText = node.group;
            document.getElementById('inspect-server').style.backgroundColor = node.color.replace(')', ', 0.15)').replace('hsl', 'hsla');
            document.getElementById('inspect-server').style.borderColor = node.color;
            document.getElementById('inspect-desc').innerText = node.description || 'No description available for this tool.';

            // Render parameter tags
            var paramContainer = document.getElementById('inspect-params');
            paramContainer.innerHTML = '';
            if (node.params && node.params.length > 0) {
                node.params.forEach(function(param) {
                    var tag = document.createElement('span');
                    tag.className = 'param-tag';
                    tag.innerText = param;
                    paramContainer.appendChild(tag);
                });
            } else {
                paramContainer.innerHTML = '<span style="font-size:12px; color:var(--color-text-secondary); font-style:italic;">None</span>';
            }

            // Render prerequisites and dependents
            var prereqs = [];
            var dependents = [];

            edges.forEach(function(edge) {
                if (edge.to === nodeId) {
                    prereqs.push({ id: edge.from, label: edge.from.split('/')[1] || edge.from, weight: edge.weight });
                }
                if (edge.from === nodeId) {
                    dependents.push({ id: edge.to, label: edge.to.split('/')[1] || edge.to, weight: edge.weight });
                }
            });

            var prereqContainer = document.getElementById('inspect-prereqs');
            prereqContainer.innerHTML = '';
            if (prereqs.length > 0) {
                prereqs.forEach(function(item) {
                    var div = document.createElement('div');
                    div.className = 'relation-item';
                    div.onclick = function() { focusOnNode(item.id); };
                    div.innerHTML = '<span>' + item.label + '</span><span class="arrow-badge">prereq</span>';
                    prereqContainer.appendChild(div);
                });
            } else {
                prereqContainer.innerHTML = '<div style="font-size:12.5px; color:var(--color-text-secondary); font-style:italic; padding:5px;">None</div>';
            }

            var dependentContainer = document.getElementById('inspect-dependents');
            dependentContainer.innerHTML = '';
            if (dependents.length > 0) {
                dependents.forEach(function(item) {
                    var div = document.createElement('div');
                    div.className = 'relation-item';
                    div.onclick = function() { focusOnNode(item.id); };
                    div.innerHTML = '<span>' + item.label + '</span><span class="arrow-badge" style="background:rgba(46, 213, 115, 0.15); color:var(--accent-emerald);">dependent</span>';
                    dependentContainer.appendChild(div);
                });
            } else {
                dependentContainer.innerHTML = '<div style="font-size:12.5px; color:var(--color-text-secondary); font-style:italic; padding:5px;">None</div>';
            }
        }

        function clearInspector() {
            document.getElementById('inspector-placeholder').style.display = 'block';
            document.getElementById('inspector-content').style.display = 'none';
        }

        // Initialize Insights Tab content
        document.getElementById('stat-total-nodes').innerText = insightsData.total_all_nodes;
        document.getElementById('stat-total-edges').innerText = insightsData.total_all_edges;

        // Render Largest Servers
        var serverList = document.getElementById('top-servers-list');
        serverList.innerHTML = '';
        insightsData.top_servers.slice(0, 5).forEach(function(item) {
            var row = document.createElement('div');
            row.className = 'insight-row';
            row.innerHTML = '<span>' + item.server + '</span><span>' + item.count + ' tools</span>';
            serverList.appendChild(row);
        });

        // Render Longest Chain
        var chainContainer = document.getElementById('longest-chain-section');
        chainContainer.innerHTML = '';
        if (insightsData.longest_chain && insightsData.longest_chain.length > 0) {
            insightsData.longest_chain.forEach(function(item, idx) {
                var step = document.createElement('div');
                step.className = 'chain-step';
                step.onclick = function() { focusOnNode(item.id); };
                step.innerHTML = '<div class="number">' + (idx + 1) + '</div><div style="flex-grow:1;"><strong>' + item.name + '</strong> <span style="font-size:10px; color:var(--color-text-secondary);">(' + item.server + ')</span></div>';
                chainContainer.appendChild(step);
            });
        } else {
            chainContainer.innerHTML = '<div style="font-size:12.5px; color:var(--color-text-secondary); font-style:italic; text-align:center; padding:10px;">No linear chain found (cycles detected)</div>';
        }

        // Render Cycles
        var cyclesContainer = document.getElementById('cycles-section');
        var cyclesCard = document.getElementById('cycles-card');
        cyclesContainer.innerHTML = '';
        if (insightsData.cycles && insightsData.cycles.length > 0) {
            cyclesCard.style.display = 'block';
            insightsData.cycles.forEach(function(cycle, cycleIdx) {
                var loopTitle = document.createElement('div');
                loopTitle.style.fontWeight = 'bold';
                loopTitle.style.fontSize = '12px';
                loopTitle.style.margin = '10px 0 5px 0';
                loopTitle.style.color = 'var(--accent-coral)';
                loopTitle.innerText = 'Loop ' + (cycleIdx + 1) + ':';
                cyclesContainer.appendChild(loopTitle);

                cycle.forEach(function(item, idx) {
                    var step = document.createElement('div');
                    step.className = 'chain-step';
                    step.onclick = function() { focusOnNode(item.id); };
                    step.innerHTML = '<div class="number" style="background:var(--accent-coral);">' + (idx + 1) + '</div><div style="flex-grow:1;"><strong>' + item.name + '</strong> <span style="font-size:10px; color:var(--color-text-secondary);">(' + item.server + ')</span></div>';
                    cyclesContainer.appendChild(step);
                });
            });
        } else {
            cyclesCard.style.display = 'none';
        }
    </script>
</body>
</html>
"""

    # Replace placeholders using safe string replacement
    html_content = html_template.replace("__NODES_DATA__", json.dumps(vis_nodes))
    html_content = html_content.replace("__EDGES_DATA__", json.dumps(vis_edges))
    html_content = html_content.replace("__INSIGHTS_DATA__", json.dumps(insights))
    html_content = html_content.replace("__TOTAL_VIS_NODES__", str(len(vis_nodes)))
    html_content = html_content.replace("__TOTAL_VIS_EDGES__", str(len(vis_edges)))
    html_content = html_content.replace("__LEGEND_ITEMS__", legend_html)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"\nInteractive HTML visualization successfully written to: {output_path}")


def main():
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    manifest_path = os.path.join(base_dir, "mcp_zero_repo/MCP-tools/mcp_tools_with_embedding.json")
    trajectory_dir = os.path.join(base_dir, "damo_convai_repo/api-bank/lv1-lv2-samples")
    cache_path = os.path.join(base_dir, "mcp_zero_repo/MCP-tools/.compose_dep_cache.json")
    output_html_path = os.path.join(base_dir, "mcp_graph.html")
    
    builder = GraphBuilder(
        trajectory_dir=trajectory_dir,
        compose_cache_path=cache_path,
        openai_api_key=None  # Rule-based heuristic or env-configured
    )
    
    G = builder.build_graph(manifest_path)
    
    extract_insights(G)
    generate_interactive_html(G, output_html_path)

if __name__ == "__main__":
    main()

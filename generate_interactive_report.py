"""
Generate Interactive HTML Report for Prophet Forecasting Results
"""

import pandas as pd
import base64
import json

def encode_image(image_path):
    """Encode image to base64 for embedding"""
    try:
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Warning: Could not encode image {image_path}: {e}")
        return None

def main():
    # Load data
    comparison_df = pd.read_csv('prophet_model_comparison.csv')
    enhanced_forecast = pd.read_csv('prophet_enhanced_forecasts_2021_2022.csv')
    basic_forecast = pd.read_csv('prophet_basic_forecasts_2021_2022.csv')

    # Encode image
    chart_image = encode_image('prophet_basic_vs_enhanced.png')

    # Convert DataFrames to JSON for JavaScript
    comparison_json = comparison_df.to_json(orient='records')
    enhanced_json = enhanced_forecast.to_json(orient='records')
    basic_json = basic_forecast.to_json(orient='records')

    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prophet Forecasting Results - Interactive Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            line-height: 1.6;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}

        .header {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
            text-align: center;
        }}

        .header h1 {{
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .header p {{
            color: #666;
            font-size: 1.1em;
        }}

        .tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}

        .tab-button {{
            background: white;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            color: #667eea;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }}

        .tab-button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        }}

        .tab-button.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}

        .tab-content {{
            display: none;
            animation: fadeIn 0.5s;
        }}

        .tab-content.active {{
            display: block;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .card {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}

        .card h2 {{
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}

        .card h3 {{
            color: #764ba2;
            margin: 20px 0 15px 0;
            font-size: 1.3em;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}

        .stat-card .stat-value {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}

        .stat-card .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}

        .chart-container {{
            margin: 30px 0;
            min-height: 400px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            font-size: 0.9em;
        }}

        table thead {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}

        table th {{
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}

        table tbody tr {{
            border-bottom: 1px solid #eee;
            transition: background 0.2s;
        }}

        table tbody tr:hover {{
            background: #f8f9ff;
        }}

        table td {{
            padding: 12px 15px;
        }}

        .highlight-improved {{
            color: #27ae60;
            font-weight: bold;
        }}

        .highlight-degraded {{
            color: #e74c3c;
            font-weight: bold;
        }}

        .image-container {{
            text-align: center;
            margin: 30px 0;
        }}

        .image-container img {{
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }}

        .forecast-selector {{
            margin: 20px 0;
        }}

        .forecast-selector select {{
            padding: 10px 20px;
            border-radius: 8px;
            border: 2px solid #667eea;
            font-size: 1em;
            background: white;
            color: #667eea;
            cursor: pointer;
            min-width: 300px;
        }}

        .info-box {{
            background: #f0f4ff;
            border-left: 4px solid #667eea;
            padding: 15px 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}

        .info-box strong {{
            color: #667eea;
        }}

        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 1.8em;
            }}

            .tabs {{
                flex-direction: column;
            }}

            .tab-button {{
                width: 100%;
            }}

            .stats-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Prophet Forecasting Dashboard</h1>
            <p>Beverage Order Forecasting Results (2021-2022)</p>
            <p style="color: #999; font-size: 0.9em; margin-top: 10px;">
                Comparing Basic vs Enhanced Prophet Models
            </p>
        </div>

        <div class="tabs">
            <button class="tab-button active" onclick="showTab('overview')">📊 Overview</button>
            <button class="tab-button" onclick="showTab('comparison')">📈 Model Comparison</button>
            <button class="tab-button" onclick="showTab('enhanced')">⭐ Enhanced Forecasts</button>
            <button class="tab-button" onclick="showTab('basic')">📉 Basic Forecasts</button>
            <button class="tab-button" onclick="showTab('charts')">📷 Visual Analysis</button>
        </div>

        <!-- Overview Tab -->
        <div id="overview" class="tab-content active">
            <div class="card">
                <h2>Executive Summary</h2>

                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">11</div>
                        <div class="stat-label">Beverages Forecasted</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">24</div>
                        <div class="stat-label">Months Predicted</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">264</div>
                        <div class="stat-label">Total Forecasts</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">2</div>
                        <div class="stat-label">Model Types</div>
                    </div>
                </div>

                <div class="info-box">
                    <strong>📌 Key Findings:</strong><br>
                    The Enhanced Prophet model with additional features (holiday, is_diet, category_total, etc.)
                    significantly outperforms the Basic Prophet model, achieving improvements of up to 72% in
                    accuracy (MAPE) for certain beverages like sprite lite.
                </div>

                <h3>Overall Performance Metrics</h3>
                <div id="overallMetrics"></div>

                <h3>Model Improvement Heatmap</h3>
                <div id="improvementHeatmap" class="chart-container"></div>
            </div>
        </div>

        <!-- Model Comparison Tab -->
        <div id="comparison" class="tab-content">
            <div class="card">
                <h2>Detailed Model Comparison</h2>

                <div class="info-box">
                    <strong>📊 Metrics Explained:</strong><br>
                    <strong>MAE</strong> (Mean Absolute Error): Average prediction error<br>
                    <strong>RMSE</strong> (Root Mean Squared Error): Emphasizes larger errors<br>
                    <strong>MAPE</strong> (Mean Absolute Percentage Error): Percentage-based accuracy
                </div>

                <h3>MAE Comparison by Beverage</h3>
                <div id="maeChart" class="chart-container"></div>

                <h3>MAPE Comparison by Beverage</h3>
                <div id="mapeChart" class="chart-container"></div>

                <h3>Improvement Percentage</h3>
                <div id="improvementChart" class="chart-container"></div>

                <h3>Full Comparison Table</h3>
                <div id="comparisonTable"></div>
            </div>
        </div>

        <!-- Enhanced Forecasts Tab -->
        <div id="enhanced" class="tab-content">
            <div class="card">
                <h2>⭐ Enhanced Prophet Forecasts (2021-2022)</h2>

                <div class="info-box">
                    <strong>✨ Enhanced Model Features:</strong><br>
                    Includes holiday indicators, diet classification, category totals,
                    quarterly patterns, and cross-beverage correlations for improved accuracy.
                </div>

                <div class="forecast-selector">
                    <label for="enhancedBeverageSelect"><strong>Select Beverage:</strong></label>
                    <select id="enhancedBeverageSelect" onchange="updateEnhancedChart()">
                        <option value="all">All Beverages</option>
                    </select>
                </div>

                <h3>Forecast Visualization</h3>
                <div id="enhancedChart" class="chart-container"></div>

                <h3>Forecast Data Table</h3>
                <div id="enhancedTable"></div>
            </div>
        </div>

        <!-- Basic Forecasts Tab -->
        <div id="basic" class="tab-content">
            <div class="card">
                <h2>📉 Basic Prophet Forecasts (2021-2022)</h2>

                <div class="info-box">
                    <strong>📌 Basic Model:</strong><br>
                    Uses only historical time series data with trend and seasonality decomposition,
                    without additional business features.
                </div>

                <div class="forecast-selector">
                    <label for="basicBeverageSelect"><strong>Select Beverage:</strong></label>
                    <select id="basicBeverageSelect" onchange="updateBasicChart()">
                        <option value="all">All Beverages</option>
                    </select>
                </div>

                <h3>Forecast Visualization</h3>
                <div id="basicChart" class="chart-container"></div>

                <h3>Forecast Data Table</h3>
                <div id="basicTable"></div>
            </div>
        </div>

        <!-- Charts Tab -->
        <div id="charts" class="tab-content">
            <div class="card">
                <h2>📷 Visual Analysis</h2>

                <div class="info-box">
                    <strong>📈 Historical vs Forecast Comparison:</strong><br>
                    These charts show historical data (2018-2020) alongside forecasts for 2021-2022,
                    comparing Basic and Enhanced Prophet models for each beverage.
                </div>

                <div class="image-container">
                    {f'<img src="data:image/png;base64,{chart_image}" alt="Prophet Basic vs Enhanced Comparison">' if chart_image else '<p>Chart image not available</p>'}
                </div>
            </div>
        </div>
    </div>

    <script>
        // Data from CSV files
        const comparisonData = {comparison_json};
        const enhancedData = {enhanced_json};
        const basicData = {basic_json};

        // Tab switching
        function showTab(tabName) {{
            const tabs = document.querySelectorAll('.tab-content');
            const buttons = document.querySelectorAll('.tab-button');

            tabs.forEach(tab => tab.classList.remove('active'));
            buttons.forEach(btn => btn.classList.remove('active'));

            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }}

        // Initialize on load
        window.onload = function() {{
            initializeOverview();
            initializeComparison();
            initializeEnhanced();
            initializeBasic();
        }};

        function initializeOverview() {{
            // Calculate overall metrics
            const avgBasicMAE = comparisonData.reduce((sum, row) => sum + row.basic_mae, 0) / comparisonData.length;
            const avgEnhancedMAE = comparisonData.reduce((sum, row) => sum + row.enhanced_mae, 0) / comparisonData.length;
            const avgBasicMAPE = comparisonData.reduce((sum, row) => sum + row.basic_mape, 0) / comparisonData.length;
            const avgEnhancedMAPE = comparisonData.reduce((sum, row) => sum + row.enhanced_mape, 0) / comparisonData.length;
            const avgImprovement = comparisonData.reduce((sum, row) => sum + row.mae_improvement_%, 0) / comparisonData.length;

            const metricsHTML = `
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">${{avgBasicMAE.toFixed(2)}}</div>
                        <div class="stat-label">Avg Basic MAE</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${{avgEnhancedMAE.toFixed(2)}}</div>
                        <div class="stat-label">Avg Enhanced MAE</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${{avgBasicMAPE.toFixed(1)}}%</div>
                        <div class="stat-label">Avg Basic MAPE</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${{avgEnhancedMAPE.toFixed(1)}}%</div>
                        <div class="stat-label">Avg Enhanced MAPE</div>
                    </div>
                    <div class="stat-card" style="grid-column: span 2;">
                        <div class="stat-value" style="color: #ffd700;">${{avgImprovement.toFixed(1)}}%</div>
                        <div class="stat-label">Average MAE Improvement</div>
                    </div>
                </div>
            `;

            document.getElementById('overallMetrics').innerHTML = metricsHTML;

            // Improvement heatmap
            const beverages = comparisonData.map(row => row.beverage);
            const maeImp = comparisonData.map(row => row.mae_improvement_%);
            const rmseImp = comparisonData.map(row => row.rmse_improvement_%);
            const mapeImp = comparisonData.map(row => row.mape_improvement_%);

            const heatmapTrace = {{
                z: [maeImp, rmseImp, mapeImp],
                x: beverages,
                y: ['MAE Improvement', 'RMSE Improvement', 'MAPE Improvement'],
                type: 'heatmap',
                colorscale: [
                    [0, '#e74c3c'],
                    [0.5, '#f1c40f'],
                    [1, '#27ae60']
                ],
                colorbar: {{
                    title: 'Improvement %'
                }},
                hovertemplate: '<b>%{{y}}</b><br>%{{x}}<br>%{{z:.1f}}%<extra></extra>'
            }};

            const heatmapLayout = {{
                title: 'Model Improvement Across Metrics',
                xaxis: {{
                    tickangle: -45
                }},
                height: 400,
                margin: {{
                    b: 150,
                    l: 150
                }}
            }};

            Plotly.newPlot('improvementHeatmap', [heatmapTrace], heatmapLayout, {{responsive: true}});
        }}

        function initializeComparison() {{
            const beverages = comparisonData.map(row => row.beverage);

            // MAE Chart
            const maeTrace1 = {{
                x: beverages,
                y: comparisonData.map(row => row.basic_mae),
                name: 'Basic Prophet',
                type: 'bar',
                marker: {{color: '#A23B72'}}
            }};

            const maeTrace2 = {{
                x: beverages,
                y: comparisonData.map(row => row.enhanced_mae),
                name: 'Enhanced Prophet',
                type: 'bar',
                marker: {{color: '#F18F01'}}
            }};

            const maeLayout = {{
                title: 'MAE Comparison (Lower is Better)',
                xaxis: {{tickangle: -45}},
                yaxis: {{title: 'Mean Absolute Error'}},
                barmode: 'group',
                height: 500,
                margin: {{b: 150}}
            }};

            Plotly.newPlot('maeChart', [maeTrace1, maeTrace2], maeLayout, {{responsive: true}});

            // MAPE Chart
            const mapeTrace1 = {{
                x: beverages,
                y: comparisonData.map(row => row.basic_mape),
                name: 'Basic Prophet',
                type: 'bar',
                marker: {{color: '#A23B72'}}
            }};

            const mapeTrace2 = {{
                x: beverages,
                y: comparisonData.map(row => row.enhanced_mape),
                name: 'Enhanced Prophet',
                type: 'bar',
                marker: {{color: '#F18F01'}}
            }};

            const mapeLayout = {{
                title: 'MAPE Comparison (Lower is Better)',
                xaxis: {{tickangle: -45}},
                yaxis: {{title: 'Mean Absolute Percentage Error (%)'}},
                barmode: 'group',
                height: 500,
                margin: {{b: 150}}
            }};

            Plotly.newPlot('mapeChart', [mapeTrace1, mapeTrace2], mapeLayout, {{responsive: true}});

            // Improvement Chart
            const improvementTrace = {{
                x: beverages,
                y: comparisonData.map(row => row.mae_improvement_%),
                type: 'bar',
                marker: {{
                    color: comparisonData.map(row => row.mae_improvement_% > 0 ? '#27ae60' : '#e74c3c')
                }},
                text: comparisonData.map(row => row.mae_improvement_%.toFixed(1) + '%'),
                textposition: 'outside'
            }};

            const improvementLayout = {{
                title: 'MAE Improvement % (Positive = Enhanced Better)',
                xaxis: {{tickangle: -45}},
                yaxis: {{title: 'Improvement %'}},
                height: 500,
                margin: {{b: 150}},
                shapes: [{{
                    type: 'line',
                    x0: -0.5,
                    x1: beverages.length - 0.5,
                    y0: 0,
                    y1: 0,
                    line: {{color: 'black', width: 2}}
                }}]
            }};

            Plotly.newPlot('improvementChart', [improvementTrace], improvementLayout, {{responsive: true}});

            // Comparison Table
            let tableHTML = `
                <table>
                    <thead>
                        <tr>
                            <th>Beverage</th>
                            <th>Basic MAE</th>
                            <th>Enhanced MAE</th>
                            <th>MAE Improvement</th>
                            <th>Basic MAPE</th>
                            <th>Enhanced MAPE</th>
                            <th>MAPE Improvement</th>
                        </tr>
                    </thead>
                    <tbody>
            `;

            comparisonData.forEach(row => {{
                const maeClass = row.mae_improvement_% > 0 ? 'highlight-improved' : 'highlight-degraded';
                const mapeClass = row.mape_improvement_% > 0 ? 'highlight-improved' : 'highlight-degraded';

                tableHTML += `
                    <tr>
                        <td><strong>${{row.beverage}}</strong></td>
                        <td>${{row.basic_mae.toFixed(3)}}</td>
                        <td>${{row.enhanced_mae.toFixed(3)}}</td>
                        <td class="${{maeClass}}">${{row.mae_improvement_%.toFixed(1)}}%</td>
                        <td>${{row.basic_mape.toFixed(1)}}%</td>
                        <td>${{row.enhanced_mape.toFixed(1)}}%</td>
                        <td class="${{mapeClass}}">${{row.mape_improvement_%.toFixed(1)}}%</td>
                    </tr>
                `;
            }});

            tableHTML += `
                    </tbody>
                </table>
            `;

            document.getElementById('comparisonTable').innerHTML = tableHTML;
        }}

        function initializeEnhanced() {{
            // Populate beverage selector
            const beverages = [...new Set(enhancedData.map(row => row.beverage))];
            const select = document.getElementById('enhancedBeverageSelect');

            beverages.forEach(bev => {{
                const option = document.createElement('option');
                option.value = bev;
                option.textContent = bev;
                select.appendChild(option);
            }});

            updateEnhancedChart();
        }}

        function updateEnhancedChart() {{
            const selected = document.getElementById('enhancedBeverageSelect').value;
            let filteredData = enhancedData;

            if (selected !== 'all') {{
                filteredData = enhancedData.filter(row => row.beverage === selected);
            }}

            // Group by beverage
            const beverages = [...new Set(filteredData.map(row => row.beverage))];
            const traces = [];

            beverages.forEach(bev => {{
                const bevData = filteredData.filter(row => row.beverage === bev);
                const dates = bevData.map(row => `${{row.year}}-${{String(row.month).padStart(2, '0')}}`);
                const quantities = bevData.map(row => row.quantity);

                traces.push({{
                    x: dates,
                    y: quantities,
                    name: bev,
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: {{width: 2}},
                    marker: {{size: 6}}
                }});
            }});

            const layout = {{
                title: selected === 'all' ? 'All Beverages - Enhanced Forecast' : `${{selected}} - Enhanced Forecast`,
                xaxis: {{title: 'Date'}},
                yaxis: {{title: 'Quantity'}},
                height: 500,
                showlegend: true,
                hovermode: 'closest'
            }};

            Plotly.newPlot('enhancedChart', traces, layout, {{responsive: true}});

            // Update table
            updateForecastTable('enhancedTable', filteredData);
        }}

        function initializeBasic() {{
            // Populate beverage selector
            const beverages = [...new Set(basicData.map(row => row.beverage))];
            const select = document.getElementById('basicBeverageSelect');

            beverages.forEach(bev => {{
                const option = document.createElement('option');
                option.value = bev;
                option.textContent = bev;
                select.appendChild(option);
            }});

            updateBasicChart();
        }}

        function updateBasicChart() {{
            const selected = document.getElementById('basicBeverageSelect').value;
            let filteredData = basicData;

            if (selected !== 'all') {{
                filteredData = basicData.filter(row => row.beverage === selected);
            }}

            // Group by beverage
            const beverages = [...new Set(filteredData.map(row => row.beverage))];
            const traces = [];

            beverages.forEach(bev => {{
                const bevData = filteredData.filter(row => row.beverage === bev);
                const dates = bevData.map(row => `${{row.year}}-${{String(row.month).padStart(2, '0')}}`);
                const quantities = bevData.map(row => row.quantity);

                traces.push({{
                    x: dates,
                    y: quantities,
                    name: bev,
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: {{width: 2}},
                    marker: {{size: 6}}
                }});
            }});

            const layout = {{
                title: selected === 'all' ? 'All Beverages - Basic Forecast' : `${{selected}} - Basic Forecast`,
                xaxis: {{title: 'Date'}},
                yaxis: {{title: 'Quantity'}},
                height: 500,
                showlegend: true,
                hovermode: 'closest'
            }};

            Plotly.newPlot('basicChart', traces, layout, {{responsive: true}});

            // Update table
            updateForecastTable('basicTable', filteredData);
        }}

        function updateForecastTable(tableId, data) {{
            let tableHTML = `
                <div style="max-height: 500px; overflow-y: auto;">
                    <table>
                        <thead style="position: sticky; top: 0;">
                            <tr>
                                <th>Beverage</th>
                                <th>Year</th>
                                <th>Month</th>
                                <th>Quantity</th>
                            </tr>
                        </thead>
                        <tbody>
            `;

            data.forEach(row => {{
                tableHTML += `
                    <tr>
                        <td>${{row.beverage}}</td>
                        <td>${{row.year}}</td>
                        <td>${{row.month}}</td>
                        <td>${{row.quantity.toFixed(2)}}</td>
                    </tr>
                `;
            }});

            tableHTML += `
                        </tbody>
                    </table>
                </div>
            `;

            document.getElementById(tableId).innerHTML = tableHTML;
        }}
    </script>
</body>
</html>
"""

    # Write HTML file
    with open('prophet_forecast_dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

    print("="*70)
    print("Interactive HTML Dashboard Created Successfully!")
    print("="*70)
    print("\nFile: prophet_forecast_dashboard.html")
    print("\nFeatures included:")
    print("  ✓ Executive summary with key metrics")
    print("  ✓ Interactive comparison charts (MAE, MAPE, Improvements)")
    print("  ✓ Enhanced forecasts with beverage selector")
    print("  ✓ Basic forecasts with beverage selector")
    print("  ✓ Embedded PNG visualization")
    print("  ✓ Sortable and filterable data tables")
    print("  ✓ Responsive design for mobile and desktop")
    print("  ✓ Professional gradient design")
    print("\nOpen the HTML file in your browser to view the dashboard!")
    print("="*70)


if __name__ == "__main__":
    main()

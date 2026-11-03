"""Plot measured generator reachability and per-board value trajectories offline."""
import json

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scripts.campaign_components_validation import OUTPUT


def main() -> None:
    root = OUTPUT / 'openai'
    rows = [json.loads(path.read_text()) for path in root.glob('value_*.json')]
    boards = sorted({(row['family'], row['board_id']) for row in rows})
    if len(boards) != 8 or len(rows) != 120:
        raise ValueError('Complete the matched validation matrix before plotting')
    figure = make_subplots(rows=4, cols=2, subplot_titles=[f'{family}: {board[-8:]}' for family, board in boards])
    colors = {'initial_examples': '#1773b1', 'trajectory_examples': '#d17b00', 'current_frozen': '#ab2a30'}
    for index, (_, board) in enumerate(boards):
        r, c = index // 2 + 1, index % 2 + 1
        for variant, color in colors.items():
            sample = sorted((item for item in rows if item['board_id'] == board and item['variant'] == variant),
                            key=lambda item: item['position'])
            last = max(item['position'] for item in sample)
            figure.add_trace(go.Scatter(x=[item['position'] / last for item in sample],
                                        y=[item['predicted_value'] for item in sample],
                                        text=[f"remaining={item['remaining']}" for item in sample],
                                        mode='lines+markers', name=variant, legendgroup=variant,
                                        showlegend=index == 0, line={'color': color}), row=r, col=c)
        sample = sorted((item for item in rows if item['board_id'] == board and item['variant'] == 'current_frozen'),
                        key=lambda item: item['position'])
        last = max(item['position'] for item in sample)
        figure.add_trace(go.Scatter(x=[item['position'] / last for item in sample],
                                    y=[item['reference_value'] for item in sample], name='witness return',
                                    legendgroup='reference', showlegend=index == 0, mode='lines+markers',
                                    line={'color': 'black', 'dash': 'dot'}), row=r, col=c)
    figure.update_layout(height=1400, title='Value on validation trajectories (missing responses appear as gaps)',
                         template='plotly_white')
    figure.update_xaxes(title_text='Position / witness length')
    figure.update_yaxes(title_text='Value', range=[0, 1.05])
    figure.write_html(str(OUTPUT / 'value_trajectories.html'), include_plotlyjs=True)
    summary = json.loads((OUTPUT / 'summary.json').read_text())
    figure2 = go.Figure(go.Bar(x=list(summary['generator']),
                              y=[row['reached_any'] / row['requests'] for row in summary['generator'].values()],
                              text=[f"{row['reached_any']}/{row['requests']}" for row in summary['generator'].values()],
                              textposition='outside'))
    figure2.update_layout(title='Generator requests with at least one CLLP-reached distinct subgoal',
                          yaxis_title='Fraction of all requests', yaxis_range=[0, 1.1], template='plotly_white')
    figure2.write_html(str(OUTPUT / 'generator_reached.html'), include_plotlyjs=True)


if __name__ == '__main__':
    main()

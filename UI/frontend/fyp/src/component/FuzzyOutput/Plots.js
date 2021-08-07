import React from 'react';
import Plot from 'react-plotly.js';

export default class Plots extends React.Component {
    render() {
        return (
            <Plot
                data={[
                    {
                        y: this.props.y_true,
                        type: 'scatter',
                        mode: 'lines+markers',
                        marker: {
                            color: 'red',
                            size: 2
                        },
                        line: {
                            width: 2
                        },
                        name: 'True'
                    },
                    {
                        y: this.props.y_pred,
                        type: 'scatter',
                        mode: 'lines+markers',
                        marker: {
                            color: 'blue',
                            size: 2
                        },
                        line: {
                            width: 2
                        },
                        name: 'Pred',
                    },
                ]}
                layout={{ width: 1200, height: 500, title: 'Accuracy Plot' }}
            />
        );
    }
}
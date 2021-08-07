import React from 'react';
import Plots from './Plots';

export default class FuzzyResult extends React.Component {
    constructor(props) {
        super(props)
        this.state = {}
    }
    render() {
        return (<div className='row align-items-center'>
            <div className='col-1'>
                <div className="row" style={{fontSize:'xx-large'}}>{this.props.score.toFixed(3)}</ div>
                <div className="row" style={{fontSize:'large'}}> Metric </ div>
            </div>
            <div className='col-11'>
                <Plots
                    y_true={this.props.y_true}
                    y_pred={this.props.y_pred} />
            </div>
        </ div>)
    }
}
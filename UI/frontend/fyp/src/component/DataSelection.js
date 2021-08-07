/* eslint-disable react/jsx-pascal-case */
import React from 'react';

import DataSet from './DataSelect/DataSet';
import DataFrame from './DataSelect/DataFrame';
import '../css/data_format.css';

export default class DataSelection extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            rerender: false,
        }
        this.toggleRender = this.toggleRender.bind(this)
        this.handleDatset = this.handleDatset.bind(this)
    }
    handleDatset=(e)=>{
        this.props.handleDataSet(e)
        this.toggleRender()
    }

    toggleRender() {
        this.setState({ rerender: !this.state.rerender })
    }

    render() {
        return (
            <>
                <div className='header1'> Fuzzy Neural Network </div>
                <div>
                    <div className='col align-self-center' >
                        <div className='row align-items-center'>
                            <div className='col align-items-left p1'>Select A Dataset</div>
                            <div className='col align-items-left'>
                                <DataSet dataset={this.props.dataset} handleDataSet={this.handleDatset} />
                            </div>
                        </ div>
                        < div className='row align-items-center'>
                            {this.props.dataset != null && this.props.dataset !== 'Iris' ?
                                <>
                                    <div className='col p1'>Select A Dataframe</div>
                                    <div className='col'>
                                        <DataFrame dataset={this.props.dataset} dataframe={this.props.dataframe}
                                            handleDataFrame={this.props.handleDataFrame}
                                            rerender={this.state.rerender}
                                            toggleRender={this.toggleRender} />
                                    </div>
                                </>
                                : null}
                        </div>
                    </ div>
                </ div>
            </>
        )
    }
}

import { Button , Spinner } from 'react-bootstrap';
import React from 'react';
import DataSelection from './DataSelection';
import FuzzyResult from './FuzzyOutput/FuzzyResult';
import * as CONSTANTS from '../constants/index';

export default class FuzzyTrain extends React.Component {
    constructor(props) {
        super(props)
        this.state = {
            dataset: null,
            dataframe: null,
            fetching: false,
            score: null,
            y_true: null,
            y_pred: null,
            i: 0
        }
        this.handleDataSet = this.handleDataSet.bind(this)
        this.handleDataFrame = this.handleDataFrame.bind(this)
        this.sendTrain = this.sendTrain.bind(this)
        this.checkButton = this.checkButton.bind(this)
        this.results = this.results.bind(this)
    }
    handleDataSet = (e) => { this.setState({ dataset: e, dataframe: null }) }
    handleDataFrame = (e) => { this.setState({ dataframe: e }) }

    checkButton() {
        if (this.state.dataframe != null || this.state.dataset === 'Iris') {
            return (
                <div className='row align-self-center'>
                    <Button variant='primary' onClick={this.sendTrain}> Train </Button>
                </ div>)
        }
        return (
            <div className='row align-self-center'>
                <Button variant='primary' disabled> Train </Button>
            </ div>)
    }

    sendTrain() {
        const axios = require('axios');
        let self = this;

        self.setState({
            fetching: true
        })

        let url = CONSTANTS.BACKENDURL + 'begin_train?dataset=' + this.state.dataset + '&dataframe=' + this.state.dataframe;
        axios.get(url).then((response) => {
            self.setState({
                fetching: false,
                score: response.data.score,
                y_true: response.data.true,
                y_pred: response.data.pred
            })
            console.log(response.data)
        }).catch((err) => {
            console.log("Error occured : ", err)
        })
    }

    results() {
        if (this.state.fetching) {
            return (<div className='row justify-content-center'>
                <Spinner animation="border" />
            </ div>)
        }
        if (this.state.score != null) {
            return (
                <FuzzyResult score={this.state.score}
                    y_true={this.state.y_true}
                    y_pred={this.state.y_pred} />
            )
        }
        else {
            return (<></>)
        }
    }

    render() {
        return (
            <>
                <DataSelection dataset={this.state.dataset}
                    dataframe={this.state.dataframe}
                    handleDataSet={this.handleDataSet}
                    handleDataFrame={this.handleDataFrame} />
                {this.checkButton()}
                {this.results()}

            </>
        )
    }
}
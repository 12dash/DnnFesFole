import React from 'react';

import * as CONSTANTS from '../../constants/index';
import { Dropdown, DropdownButton } from 'react-bootstrap';

export default class DataFrame extends React.Component {
    constructor(props) {
        super(props)
        this.state = {
            dataframe_options: null,
            fetching: false
        }
        this.fetch = this.fetch.bind(this)
    }

    fetch() {
        var self = this
        if (!self.state.fetching) {
            self.setState({ fetching: true })
            const axios = require('axios');
            var url = CONSTANTS.BACKENDURL + 'get_dirs?dataset=' + this.props.dataset
            axios.get(url).then((response) => {
                self.setState({ dataframe_options: response.data.paths, fetching: false })
            }).catch((error) => {
                console.log(error);
            })
        }
    }

    render() {
        let title = this.props.dataframe == null ? 'Dataframe Options' : this.props.dataframe
        if (this.props.rerender){
            this.props.toggleRender()
            this.fetch()
        }
        if (this.state.dataframe_options != null)
            return (
                <DropdownButton title={title} onSelect={this.props.handleDataFrame}>
                    {this.state.dataframe_options.map((value) => {
                        return (<Dropdown.Item eventKey={value}>{value}</Dropdown.Item>)
                    })}
                </ DropdownButton >
            )
        else {
            this.fetch()
            return (<></>)
        }

    }
}
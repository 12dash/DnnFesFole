import React from 'react';
import { Dropdown, DropdownButton } from 'react-bootstrap';

export default class DataSet extends React.Component {
    constructor(props) {
        super(props)
        this.state = {
            dataset_options: ['Traffic']
        }
    }    

    render() {
        let title = this.props.dataset === null ? 'DataSet Options' :  this.props.dataset
        return (
            <>
                <DropdownButton title={title} onSelect={this.props.handleDataSet}>
                    {this.state.dataset_options.map((value) => {
                        return(<Dropdown.Item eventKey={value}>{value}</Dropdown.Item>)
                    })}
                </ DropdownButton >
            </>
        )
    }
}
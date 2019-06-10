import React from "react";

import { Link } from 'react-router-dom';
import { BrowserRouter as Router, Route } from 'react-router-dom';


/*var $ = require('jquery');*/
require('../css/App.css');

import CustomNavbar from './components/CustomNavbar';
import Home from './components/Home';

class App extends React.Component {

    constructor(props) {
        super(props);
    }

    render () {

        return (
          <Router>
            <div>
                <CustomNavbar />
                <Route exact path="/" component={Home} />
            </div>
          </Router>
          );
    }
}

export default App;


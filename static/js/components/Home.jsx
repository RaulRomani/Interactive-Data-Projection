import React from "react";
import ReactDOM from "react-dom"; //import from a global dependency
import serializeForm from 'form-serialize'


//var $ = require('jquery');
require('../../css/Home.css');
import { Link } from 'react-router-dom'

import * as DR_API from './utils/DR_API'
import PlotPoints from "./PlotPoints"; //import from a local file
import PlotComparison from "./PlotComparison"; //import from a local file


//import SelectParameters from './SelectParameters';
function randomDataSet(dataSetSize, minValue, maxValue) {
      return new Array(dataSetSize).fill(0).map(function(n) {
        return Math.random() * (maxValue - minValue) + minValue;
      });
    }


export default class Home extends React.Component {
    constructor(props) {
      super(props);
      this.selectControlPoints = this.selectControlPoints.bind(this);
      this.projectUsingEnsemble = this.projectUsingEnsemble.bind(this);
      this.estimateEnsemble = this.estimateEnsemble.bind(this);


      this.showLampWeight = this.showLampWeight.bind(this);
      this.showLspWeight = this.showLspWeight.bind(this);
      this.showPlmpWeight = this.showPlmpWeight.bind(this);

      
      this.state = {controlPoints: {}, datasetName : "", ensembleProjection: {}}
    }

    showLampWeight(val){ document.getElementById('lamp_weight_val').innerHTML = val}
    showLspWeight(val){ document.getElementById('lsp_weight_val').innerHTML = val}
    showPlmpWeight(val){ document.getElementById('plmp_weight_val').innerHTML = val}

    selectControlPoints (e) {
      e.preventDefault()
      //console.log(e.target.elements)
      const values = serializeForm(e.target, { hash: true })
      console.log(values)

      DR_API.getControlPoints(values).then((xy) => {
        this.setState({ controlPoints : xy, datasetName: values["datasetName"] })

        ReactDOM.render(<PlotPoints controlPoints={this.state.controlPoints}  tittle = "Projected control points" />, document.getElementsByClassName("plot-control-points")[0])

      })
    }

    projectUsingEnsemble (e) {
      e.preventDefault()
      //console.log(e.target.elements)
      const values = serializeForm(e.target, { hash: true })

      console.log(values)

      let data = {values: values, controlPoints: this.state.controlPoints, datasetName: this.state.datasetName }

      console.log("parameters passed")
      console.log(data)

      DR_API.projectUsingEnsemble(data).then((data) => {
        console.log("output")
        console.log(data)

        /*TODO: Create a plot metrics component */
        /*TODO: Undate the plot metrics*/

        //<PlotComparison methods={} tittle={} />
        let ensembleProjection = data.projections
        let metrics = data.metrics

        ReactDOM.render(<PlotComparison methods = {metrics.NP} tittle = "Neighborhood preservation"  />, document.getElementsByClassName("NP")[0])
        ReactDOM.render(<PlotComparison methods = {metrics.T} tittle = "Trustworthiness"  />, document.getElementsByClassName("T")[0])
        ReactDOM.render(<PlotComparison methods = {metrics.NH} tittle = "Neighborhood hit"  />, document.getElementsByClassName("NH")[0])





        this.setState({ ensembleProjection : ensembleProjection })
        ReactDOM.render(<PlotPoints controlPoints={this.state.controlPoints} LAMP={ensembleProjection.LAMP} tittle = "LAMP"  />, document.getElementsByClassName("LAMP")[0])
        ReactDOM.render(<PlotPoints controlPoints={this.state.controlPoints} LSP={ensembleProjection.LSP} tittle = "LSP"  />, document.getElementsByClassName("LSP")[0])
        ReactDOM.render(<PlotPoints controlPoints={this.state.controlPoints} PLMP={ensembleProjection.PLMP} tittle = "PLMP"  />, document.getElementsByClassName("PLMP")[0])
        ReactDOM.render(<PlotPoints controlPoints={this.state.controlPoints} PLMP={ensembleProjection.Ensemble} tittle = "Ensemble"  />, document.getElementsByClassName("Ensemble")[0])

      })
    }

    estimateEnsemble (e) {
      e.preventDefault()
      console.log("estimating ensemble")
      console.log(this.state)

      let data = {Ensemble: this.state.ensembleProjection.Ensemble,  datasetName: this.state.datasetName }
      console.log(data)

      DR_API.estimateEnsemble(data).then((estimate) => {
        console.log("estimate")
        console.log(estimate)

        /*TODO: Create a plot metrics component */
        /*TODO: Add Estimate_metrics to this.state.metrics*/
        /*TODO: Undate the plot metrics*/
        ReactDOM.render(<PlotPoints controlPoints={this.state.controlPoints} LSP={estimate} tittle = "Estimate"  />, document.getElementsByClassName("Estimate")[0])

        /*this.setState({ ensembleProjection : ensembleProjection })*/
        /*ReactDOM.render(<PlotPoints controlPoints={this.state.controlPoints} LAMP={ensembleProjection.LAMP} tittle = "LAMP"  />, document.getElementsByClassName("LAMP")[0])
        ReactDOM.render(<PlotPoints controlPoints={this.state.controlPoints} LSP={ensembleProjection.LSP} tittle = "LSP"  />, document.getElementsByClassName("LSP")[0])
        ReactDOM.render(<PlotPoints controlPoints={this.state.controlPoints} PLMP={ensembleProjection.PLMP} tittle = "PLMP"  />, document.getElementsByClassName("PLMP")[0])
        ReactDOM.render(<PlotPoints controlPoints={this.state.controlPoints} PLMP={ensembleProjection.Ensemble} tittle = "Ensemble"  />, document.getElementsByClassName("Ensemble")[0])*/

      })
    }

    
   
    render () {
      console.log("All good")

      return (

        <div class = "container">
          <div class="section">
            <div class="row">
              <div class="col s12 m4">

                <div class="card card-dashboard">
                  <div class="card-content "> 
                    <span class="card-title">Project control points</span>
                  </div>
                  <div className="card-action">
                    <form onSubmit={this.selectControlPoints}>
                      <div class="input-field col s12">
                        <select name="datasetName">
                          <option value="" disabled selected>options</option>
                          <option>Synthetic4Classes</option>
                          <option>AustralianCA</option>
                          <option>Dermatology</option>
                          <option selected="true" >Iris</option>
                        </select>
                        <label>Select dataset name</label>
                      </div>
                      <div className="center-align">
                        <button class="btn waves-effect waves-light" type="submit" name="action">Project CP</button>
                      </div>
                    </form>
                  </div>
                </div>

                <div class="card card-dashboard">
                  <div class="card-content "> 
                    <span class="card-title">Project using ensemble</span>
                  </div>
                  <div className="card-action">

                    <form onSubmit={this.projectUsingEnsemble}>
                      <div class="row">
                        <div class="col s12 m4">
                          <label>
                            <input type="checkbox" name="methods" value="LAMP" checked="checked" />
                            <span>LAMP</span>
                          </label>
                        </div>
                        <div class="col s12 m6">
                          <div class="range-field">
                            <input type="range"  min="0" max="1" step="0.1" name="LAMP_weight" onChange={(event) => this.showLampWeight(event.target.value)} />
                          </div>
                          {/*<input placeholder="weight"  min={0} max={1} type="range" step="0.1" class="validate input__weight"/>*/}
                        </div>
                        <div class="col s12 m2">
                          <span id="lamp_weight_val">1</span>
                        </div>

                        <div class="col s12 m4">
                          <label>
                            <input type="checkbox" name="methods" value="LSP" checked="checked" />
                            <span>LSP</span>
                          </label>
                        </div>
                        <div class="col s12 m6">
                          <div class="range-field">
                            <input type="range"  min="0" max="1" step="0.1" name="LSP_weight" onChange={(event) => this.showLspWeight(event.target.value)} />
                          </div>
                        </div>
                        <div class="col s12 m2">
                          <span id="lsp_weight_val">1</span>
                        </div>

                        <div class="col s12 m4">
                          <label>
                            <input type="checkbox" name="methods" value="PLMP" checked="checked"/>
                            <span>PLMP</span>
                          </label>
                        </div>
                        <div class="col s12 m6">
                          <div class="range-field">
                            <input type="range"  min="0" max="1" step="0.1" name="PLMP_weight" onChange={(event) => this.showPlmpWeight(event.target.value)} />
                          </div>
                        </div>
                        <div class="col s12 m2">
                          <span id="plmp_weight_val">1</span>
                        </div>

                      </div>


                      <div className="center-align">
                        <button class="btn waves-effect waves-light" type="submit" name="action">Project</button>
                      </div>
                    </form>

                  </div>
                </div>

                <div class="card card-dashboard">
                  <div class="card-content "> 
                    <span class="card-title">Estimate model with DL</span>
                  </div>
                  <div className="card-action">
                    <form onSubmit={this.estimateEnsemble}>
                      {/*<div class="input-field col s12">
                        <select name="datasetName">
                          <option value="" disabled selected>options</option>
                          <option>Synthetic4Classes</option>
                          <option>AustralianCA</option>
                          <option>Dermatology</option>
                          <option selected="true" >Iris</option>
                        </select>
                        <label>Select dataset name</label>
                      </div>*/}
                      <div className="center-align">
                        <button class="btn waves-effect waves-light" type="submit" name="action">Estimate</button>
                      </div>
                    </form>
                  </div>
                </div>
                  
              </div>
              <div class="col s12 m8">
                <div className="plot-control-points" ></div>
              </div>
            </div>
            
            <div class="row">
              <div class="col s12 m7">
                <ul class="tabs tabs__projections">
                  <li class="tab col s2"><a class="active" href="#LAMP">LAMP</a></li>
                  <li class="tab col s2"><a href="#LSP">LSP</a></li>
                  <li class="tab col s2"><a href="#PLMP">PLMP</a></li>
                  <li class="tab col s2"><a href="#ensemble">Ensemble</a></li>
                  <li class="tab col s2"><a href="#estimate">Estimate</a></li>
                </ul>
                <div id="LAMP"><div className="LAMP" ></div></div>
                <div id="LSP"><div className="LSP" ></div></div>
                <div id="PLMP"><div className="PLMP" ></div></div>
                <div id="ensemble" >
                  <div className="Ensemble" ></div>
                </div>
                <div id="estimate">
                  <div className="Estimate" ></div>
                </div>
              </div>
              <div class="col s12 m5" >

                <ul class="tabs tabs__projections">
                  <li class="tab col s2"><a class="active" href="#NP">NP</a></li>
                  <li class="tab col s2"><a href="#T">T</a></li>
                  <li class="tab col s2"><a href="#NH">NH</a></li>
                </ul>
                <div id="NP"><div className="NP" ></div></div>
                <div id="T"><div className="T" ></div></div>
                <div id="NH"><div className="NH" ></div></div>

              </div>
              
              
            </div>
            
          </div>
        </div>
      	
        );
    }
}

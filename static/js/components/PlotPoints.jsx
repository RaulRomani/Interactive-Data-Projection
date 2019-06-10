import React from "react";
import ReactDOM from "react-dom"; //import from a global dependency

import serializeForm from 'form-serialize'

import Plot from 'react-plotly.js';
import { Link } from 'react-router-dom'

function randomDataSet(dataSetSize, minValue, maxValue) {
  return new Array(dataSetSize).fill(0).map(function(n) {
    return Math.random() * (maxValue - minValue) + minValue;
  });
}


export default class Home extends React.Component {
    constructor(props) {
      super(props);
    }

    render () {
      const { controlPoints, ensembleProjection, LAMP, LSP, PLMP, tittle } = this.props
      //const { query } = this.state

      let plots = []

      if (typeof controlPoints !== 'undefined'){
        plots.push( {
                      name: 'Control points',
                      x: controlPoints.x ,
                      y: controlPoints.y,
                      type: "scattergl",
                      mode: 'markers',
                      opacity: 0.5,

                      marker: {color: controlPoints.labels,
                               colorscale: 'Viridis' ,
                               size:12,
                               editable: true}
                    })

      } else {
        /*console.log("ensembleProjection is undefined")*/
      }
      
      
      if (typeof ensembleProjection !== 'undefined'){
        plots.push( {
                      name: 'projection',
                      x: ensembleProjection.x ,
                      y: ensembleProjection.y,
                      type: "scattergl",
                      mode: 'markers',
                      opacity: 0.7,
                      marker: {color: ensembleProjection.labels,
                               colorscale: 'Viridis' ,
                               size:8,}
                    })

      } else {
        /*console.log("ensembleProjection is undefined")*/
      }

      if (typeof LAMP !== 'undefined'){
        plots.push( {
                      name: 'projection',
                      x: LAMP.x ,
                      y: LAMP.y,
                      type: "scattergl",
                      mode: 'markers',
                      opacity: 0.7,
                      marker: {color: LAMP.labels,
                               colorscale: 'Viridis' ,
                               size:8,}
                    })

      } else {
        /*console.log("ensembleProjection is undefined")*/
      }
      if (typeof LSP !== 'undefined'){
        plots.push( {
                      name: 'projection',
                      x: LSP.x ,
                      y: LSP.y,
                      type: "scattergl",
                      mode: 'markers',
                      opacity: 0.7, 
                      marker: {color: LSP.labels,
                               colorscale: 'Viridis' ,
                               size:8,}
                    })

      } else {
        /*console.log("ensembleProjection is undefined")*/
      }

      if (typeof PLMP !== 'undefined'){
        plots.push( {
                      name: 'Ensemble projection',
                      x: PLMP.x ,
                      y: PLMP.y,
                      type: "scattergl",
                      mode: 'markers',
                      opacity: 0.7, 
                      marker: {color: PLMP.labels,
                               colorscale: 'Viridis' ,
                               size:8,}
                    })

      } else {
        /*console.log("ensembleProjection is undefined")*/
      }





      return (
              <Plot
                  data={plots}
                  layout={{width: 600, height: 600, title: tittle}}
                  config={{ 
                          }}

              />
        );
    }
}
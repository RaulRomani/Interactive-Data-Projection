import React from "react";
import Plot from 'react-plotly.js';


export default class Home extends React.Component {
    constructor(props) {
      super(props);
    }
    
   
    render () {
      const { methods, tittle } = this.props

      var x_values = [5,10,15,20,25,30,35,40]
      var color_palette = ['rgba(37, 120, 176,1)', 'rgba(254, 125, 41,1)', 'rgba(53, 157, 59,1)', 'rgba(211, 43, 46,1)', '#9269B8', '#8A574D', '#E178BE', '#969696', '#BBBA3B', '#29BECC', '#F97E74', '#80AFCD', '#FBAE6E', '#B3D879', '#FACBE1']


      let scatters_line = []
      for (let k = 0; k < methods.length; k++) {
        /*var results = methods.values.map( (result) => result.value)*/
        scatters_line.push({type: 'scatter', x: x_values,  y: methods[k].values, name: methods[k].methodName, marker:{color: color_palette[k]} })

        
      }
      console.log(scatters_line)



      return (<Plot key={0} data={scatters_line}  className="metric_plot"
                    layout={{width: 500,  
                             title: tittle, 
                             xaxis: { dtick :5}
                             }}
                           
                     />
              );
    }
}








                    
import React from "react";
import ReactDOM from "react-dom"; //import from a global dependency
import App from "./App"; //import from a local file
/*import 'bootstrap/dist/css/bootstrap.css';*/

/*
console.log("index.jsx")

var $ = require('jquery');

console.log("wft1!")
$(document).ready(function(){
  console.log("wft!")
  $('select').formSelect();
});*/




// what to render and where to render it
ReactDOM.render(<App />, document.getElementById("content"));


# Interactive data projection webapp
Interactive web application to do data projection using three methods, [LAMP](https://www.researchgate.net/profile/Fernando_Paulovich/publication/220668290_Local_Affine_Multidimensional_Projection/links/578e4ef008ae81b4466ec19c/Local-Affine-Multidimensional-Projection.pdf), [LSP](https://www.researchgate.net/profile/Rosane_Minghim/publication/5483798_Least_Square_Projection_A_Fast_High-Precision_Multidimensional_Projection_Technique_and_Its_Application_to_Document_Mapping/links/02bfe510ac4c4e370f000000.pdf), and [PLMP](https://www.sci.utah.edu/publications/Pau2010a/Paulovich_TVCG2010.pdf), using a linear combination of the methods using weights. Project build with [Flask](http://flask.pocoo.org/), [React](https://reactjs.org/) and [Plotly](https://plot.ly/)

# Installation

### Install node modules and build the app

```
cd static
npm install
npm run build
```

### Run the server.py

```
cd server
python server.py
```

### Open your browser and paste this url

```
http://localhost:5000/
```

### Screenshot

![Alt text](screenshot/1.png)
![Alt text](screenshot/2.png)

### License

MIT
const webpack = require('webpack');
var ExtractTextPlugin = require('extract-text-webpack-plugin');

const config = {
    /* entry: { "indexPage": __dirname + "js/index.jsx",
                "aboutPage": __dirname + "js/about.jsx"} */
    entry:  __dirname + '/js/index.jsx',
    output: {
        path: __dirname + '/dist',
        filename: 'bundle.js', // "[name].js"
    },
    resolve: {
        extensions: [".js", ".jsx", ".css"]
    },
    module: { // add a babel-loader rule 
        rules: [
            {
                test: /\.jsx?/,
                exclude: /node_modules/,
                use: 'babel-loader'
            },
            {
                test: /\.css$/,
                use: ExtractTextPlugin.extract({
                        fallback: 'style-loader',
                        use: 'css-loader',
                     })
            }
            ,{
                test: /\.(png|svg|jpg|gif)$/,
                //use: 'file-loader?outputPath=images/'
                use: 'file-loader?name=[name].[ext]'

                /*use: [{
                    loader: 'file-loader',
                    options: {
                        name: '[name].[ext]',
                        outputPath: 'img/',
                        publicPath: 'img/'

                    }
                }]*/
            }
        ]
    },
    plugins: [
        //new ExtractTextlugin('styles.css'),
        new ExtractTextPlugin({
          filename: 'styles.css',
          disable: false,
        }),

        new webpack.optimize.UglifyJsPlugin({
          include: /\.min\.js$/,
          minimize: true
        }),
        new webpack.DefinePlugin({
                 'process.env.NODE_ENV': '"production"'
        }),

        /*new webpack.HotModuleReplacementPlugin(),
        new webpack.NamedModulesPlugin(),*/
    ],
    /*devServer: {
        hot:true,
        open:true,
        proxy: {
        "/": "http://localhost:5000"
        }
    }*/
};

module.exports = config;
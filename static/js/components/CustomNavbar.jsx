import  { Component }  from 'react'

export default class CustomNavbar extends Component {

  constructor(props) {
    super(props);

    this.toggle = this.toggle.bind(this);
    this.state = {
      isOpen: false
    };
  }
  toggle() {
    this.setState({
      isOpen: !this.state.isOpen
    });
  }

  render() {
    return (
      <nav class="light-blue lighten-1">
        <div class="container nav-wrapper">
          <a href="#" class="brand-logo">WebApp</a>
          <ul id="nav-mobile" class="right hide-on-med-and-down">
            <li><a href="#">Options</a></li>
          </ul>
        </div>
      </nav>
    )
  }
}

const React = require("react");
const ReactMarkdown = require("react-markdown");
const render = require("react-dom").render;
const gfm = require("remark-gfm");

const markdown = `Just a link: https://reactjs.com.`;

render(<ReactMarkdown plugins={[gfm]} children={markdown} />, document.body);
console.log("HELLO REACT!!!");

// const e = React.createElement;

// class LikeButton extends React.Component {
//   constructor(props) {
//     super(props);
//     this.state = { liked: false };
//   }

//   render() {
//     if (this.state.liked) {
//       return "You liked this.";
//     }

//     return e(
//       "button",
//       { onClick: () => this.setState({ liked: true }) },
//       "Like"
//     );
//   }
// }
// const domContainer = document.querySelector("#like_button_container");
// ReactDOM.render(e(LikeButton), domContainer);

import './App.css';
import { BrowserRouter as Router, Switch, Route } from "react-router-dom";
import LandingPage from './component/LandingPage';
import FuzzyTrain from './component/FuzzyTrain';

function App() {
  return (
    <div className='container'>
      <Router>
        <Switch>
          <Route exact path='/'> <LandingPage /></Route>
          <Route path='/fuzzy'><FuzzyTrain /></Route>
        </Switch>
      </Router>
    </div>
  );
}

export default App;

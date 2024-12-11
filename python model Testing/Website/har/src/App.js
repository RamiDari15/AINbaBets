import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Login from "./Pages/Login";
import CreateAccount from "./Pages/CreateAccount";
import Parlay from "./Pages/Parlay";
import AllParlays from "./Pages/AllLegs";

const App = () => {
  return (
    <Router>
        <Routes>
          <Route path="/" element={<Login />} />
          <Route path="/create-profile" element={<CreateAccount/>} />
          <Route path="/parlay" element={<Parlay />} />
          <Route path="/AllParlays" element={<AllParlays />} />
        </Routes>
    </Router>
  );
};

export default App;

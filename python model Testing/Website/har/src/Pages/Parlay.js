import React, { useState, useEffect } from "react";
import axios from "axios";
import "../Css/Parlay.css";
import Select from "react-select";
import { useNavigate } from "react-router-dom";
import { doc, updateDoc, arrayUnion } from "firebase/firestore";
import { auth, firestore } from "./firebaseConfig"; // Ensure Firestore is exported

const ParlayPage = () => {
  const [legs, setLegs] = useState([]);
  const [players, setPlayers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [parlayResult, setParlayResult] = useState(null);

  const navigate = useNavigate();

  const handleSignOut = () => {
    localStorage.removeItem("userToken"); 
    navigate("/");
  };


  const addLeg = () => {
    setLegs([...legs, { player: "", stat: "points", overUnder: "over", value: "" }]);
  };

  const updateLeg = (index, key, value) => {
    const newLegs = [...legs];
    newLegs[index][key] = value;
    setLegs(newLegs);
  };

  const removeLeg = (index) => {
    const newLegs = legs.filter((_, legIndex) => legIndex !== index);
    setLegs(newLegs);
  };

  const resetLegs = () => {
    setLegs([]);
    setParlayResult(null);
  };



  
  const handleSubmit = async () => {
    try {
      const response = await axios.post("http://localhost:5002/parlay/calculate", { legs });
      const { parlayPercentage } = response.data ; 
      const percent = parlayPercentage * 100;
  
      if (parlayPercentage !== undefined) {
        setParlayResult(percent.toFixed(2));


        const currentUser = auth.currentUser;
        if (currentUser) {
          const userId = currentUser.uid;
  

          const newParlay = {
            legs, 
            percentage: percent, 
            createdAt: new Date().toISOString(), 
          };
  
          
          await updateDoc(doc(firestore, "Parlays", userId), {
            Parlays: arrayUnion(newParlay),
          });
  
          console.log("Parlay added successfully to Firestore.");
        } else {
          console.error("No user is currently signed in.");
        }
      } else {
        console.error("parlayPercentage not found in the response");
        setParlayResult(null);
      }
    } catch (error) {
      console.error("Error submitting parlay:", error);
      setParlayResult(null);
    }
  };
  
    
  

  const fetchPlayers = async () => {
    setLoading(true);
    try {
      const response = await axios.get("http://localhost:5002/players/players");
      setPlayers(response.data);
      console.log(response)
    } catch (error) {
      console.error("Error fetching players:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleSeeAllParlays = () => {
    navigate("/AllParlays"); 
  };

  

  useEffect(() => {
    fetchPlayers();
  }, []);

  return (
    <div className="parlay-page">
      <nav className="navbar">
        <button className="sign-out" onClick={handleSignOut}>Sign Out</button>
        <button className="see-all-parlays" onClick={handleSeeAllParlays}>
      See All Parlays
    </button>
      </nav>

      <h1 className="title">NBA Parlay Predictor</h1>
      <p className="date">{new Date().toLocaleDateString("en-US", { month: "numeric", day: "numeric", year: "numeric" })}</p>

      {parlayResult !== null && (
  <div className={`parlay-result ${parlayResult < 50 ? "warning" : "success"}`}>
    <h2>Parlay Result</h2>
    <p>{parlayResult}%</p>
  </div>
)}


      {loading ? (
        <p>Loading players...</p>
      ) : (
        <div className="legs">
          {legs.map((leg, index) => (
            <div key={index} className="leg">
              <label>
                Player:

                <Select
  isSearchable
  value={
    leg.player
      ? {
          value: leg.player,
          label: players.find((player) => player.player_id === leg.player)?.name || "",
        }
      : null
        }
        onChange={(selectedPlayer) => {
            const selected = players.find(
            (player) => player.player_id === selectedPlayer.value
            );
            if (selected) {
            updateLeg(index, "player", selected.player_id);
            updateLeg(index, "team_id", selected.team_id);
            updateLeg(index, "opponent_team_id", selected.opponent_team_id);
            }
        }}
        options={players.map((player) => ({
            value: player.player_id,
            label: player.name,
            team_id: player.team_id,
            opponent_team_id: player.opponent_team_id,
        }))}
        placeholder="Search or select player"
        />
                    </label>

              <label>
                Stat:
                <select
                  value={leg.stat}
                  onChange={(e) => updateLeg(index, "stat", e.target.value)}
                >
                  <option value="points">Points</option>
                  <option value="rebounds">Rebounds</option>
                  <option value="assists">Assists</option>
                  <option value="blocks">Blocks</option>
                  <option value="steals">Steals</option>
                </select>
              </label>

              <label>
                Over/Under:
                <select
                  value={leg.overUnder}
                  onChange={(e) => updateLeg(index, "overUnder", e.target.value)}
                >
                  <option value="over">Over</option>
                  <option value="under">Under</option>
                </select>
              </label>

              <label>
                Value:
                <input
                  type="number"
                  value={leg.value}
                  onChange={(e) => updateLeg(index, "value", e.target.value)}
                />
              </label>

              <button className="remove-leg" onClick={() => removeLeg(index)}>
                Remove Leg
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="buttons">
        <button className="add-leg" onClick={addLeg}>
          Add Leg
        </button>
        <button className="reset-legs" onClick={resetLegs}>
          Reset All Legs
        </button>
        <button className="submit-parlay" onClick={handleSubmit}>
          Submit Parlay
        </button>
      </div>
    </div>
  );
};

export default ParlayPage;

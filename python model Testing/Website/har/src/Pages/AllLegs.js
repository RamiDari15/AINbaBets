import React, { useEffect, useState } from "react";
import { doc, getDoc } from "firebase/firestore";
import { auth, firestore } from "./firebaseConfig";
import "../Css/AllParlays.css";

const AllParlays = () => {
  const [parlays, setParlays] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchParlays = async () => {
      try {
        const currentUser = auth.currentUser;
        if (!currentUser) {
          setError("No user is currently signed in.");
          setLoading(false);
          return;
        }

        const userId = currentUser.uid;
        const docRef = doc(firestore, "Parlays", userId);
        const docSnap = await getDoc(docRef);

        if (docSnap.exists()) {
          const data = docSnap.data();
          console.log("Fetched Parlays:", data.Parlays);
          setParlays(data.Parlays || []);
        } else {
          setError("No parlays found for this user.");
        }
      } catch (err) {
        console.error("Error fetching parlays:", err);
        setError("Failed to fetch parlays. Please try again.");
      } finally {
        setLoading(false);
      }
    };

    fetchParlays();
  }, []);

  if (loading) return <p>Loading...</p>;
  if (error) return <p className="error-message">{error}</p>;

  return (
    <div className="all-parlays-container">
      <h2>All Parlays</h2>
      {parlays.length === 0 ? (
        <p>No parlays to display.</p>
      ) : (
        <div className="parlays-list">
          {parlays.map((parlay, index) => (
            <div key={index} className="parlay-item">
              <h3>Parlay {index + 1}</h3>
              <p>
                <strong>Percentage:</strong> {(parlay.percentage * 100).toFixed(2)}%
              </p>
              <div>
                <strong>Legs:</strong>
                <ul>
                  {parlay.legs.map((leg, legIndex) => (
                    <li key={legIndex}>
                      {typeof leg === "object"
                        ? `Stat: ${leg.stat}, Player: ${leg.player}, Value: ${leg.value}`
                        : leg}
                    </li>
                  ))}
                </ul>
              </div>
              <p>
                <strong>Created At:</strong> {parlay.createdAt ? new Date(parlay.createdAt).toLocaleString() : "N/A"}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default AllParlays;

import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { getAuth, createUserWithEmailAndPassword } from "firebase/auth";
import { getFirestore, doc, setDoc } from "firebase/firestore";

const CreateProfilePage = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();
  const firestore = getFirestore();

  const handleCreateAccount = async (e) => {
    e.preventDefault();
    const auth = getAuth();

    try { 
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      const userId = userCredential.user.uid;

      await setDoc(doc(firestore, "users", userId), {
        userId: userId,
        parlays: []
      });

      navigate("/login"); 
    } catch (err) {
      setError("Failed to create an account. Please try again.");
    }
  };

  return (
    <div>
      <h2>Create Profile</h2>
      <form onSubmit={handleCreateAccount}>
        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        <button type="submit">Create Account</button>
      </form>
      {error && <p>{error}</p>}
    </div>
  );
};

export default CreateProfilePage;

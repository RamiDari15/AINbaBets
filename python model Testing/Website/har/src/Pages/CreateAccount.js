import React, { useState } from "react";
import { createUserWithEmailAndPassword, sendEmailVerification } from "firebase/auth";
import { doc, setDoc } from "firebase/firestore"; // Import Firestore functions
import { useNavigate } from "react-router-dom";
import { auth, firestore } from "./firebaseConfig"; // Ensure Firestore is exported
import "../Css/Create.css";

const CreateAccount = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const navigate = useNavigate();

  const handleCreateAccount = async (e) => {
    e.preventDefault();
    try {

      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      await sendEmailVerification(userCredential.user);

     
      const userId = userCredential.user.uid;
      await setDoc(doc(firestore, "Parlays", userId), {
        email: email, 
        Parlays: [],
      });

      setSuccess("Account created successfully. Please verify your email.");
      setError("");
    } catch (error) {
      console.error("Error creating account: ", error);
      setError("Failed to create account. Please try again.");
      setSuccess("");
    }
  };

  return (
    <div className="create-account-container">
      <h2 className="create-account-title">Create Account</h2>
      <form className="create-account-form" onSubmit={handleCreateAccount}>
        <input
          type="email"
          className="create-account-input"
          placeholder="Enter your email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
        <input
          type="password"
          className="create-account-input"
          placeholder="Enter your password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        <button type="submit" className="create-account-button">
          Create Account
        </button>
      </form>
      {error && <p className="error-message">{error}</p>}
      {success && <p className="success-message">{success}</p>}

      <button
        className="back-to-login-button"
        onClick={() => navigate("/")} // Navigate back to login
      >
        Go Back to Login
      </button>
    </div>
  );
};

export default CreateAccount;

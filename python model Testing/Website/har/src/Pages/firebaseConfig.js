import { initializeApp } from "firebase/app";
import { getAuth, sendEmailVerification } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
    apiKey: "AIzaSyAtA3ec0rZrLvCaId2-gOWTqTtiteVyDC0",
    authDomain: "harr-a8946.firebaseapp.com",
    projectId: "harr-a8946",
    storageBucket: "harr-a8946.firebasestorage.app",
    messagingSenderId: "587566219942",
    appId: "1:587566219942:web:b7fe492a6a330663c4ea67",
    measurementId: "G-ZP5K23TE1P"
  };
  
  const app = initializeApp(firebaseConfig);
  const auth = getAuth(app);
  
  const firestore = getFirestore(app);

  export { auth, firestore };
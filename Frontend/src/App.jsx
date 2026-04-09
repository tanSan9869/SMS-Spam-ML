import React, { useState } from "react";
import axios from "axios";

function App() {
  const [message, setMessage] = useState("");
  const [result, setResult] = useState("");
  const [modelAccuracy, setModelAccuracy] = useState(null);

  const handleSubmit = async () => {
    try {
      const res = await axios.post("http://127.0.0.1:5000/predict", {
        message: message,
      });
      setResult(res.data.prediction);
      setModelAccuracy(res.data.model_accuracy);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h1>Spam Detection</h1>

      <textarea
        rows="5"
        cols="40"
        placeholder="Enter message..."
        value={message}
        onChange={(e) => setMessage(e.target.value)}
      />

      <br /><br />

      <button onClick={handleSubmit}>Check</button>

      <h2>Result: {result}</h2>
      {modelAccuracy !== null && (
        <p>Model Accuracy: {(modelAccuracy * 100).toFixed(2)}%</p>
      )}
    </div>
  );
}

export default App;
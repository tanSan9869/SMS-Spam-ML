import React, { useState } from "react";
import axios from "axios";

function App() {
  const [message, setMessage] = useState("");
  const [result, setResult] = useState("");
  const [modelAccuracy, setModelAccuracy] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    if (!message.trim()) return;

    setLoading(true);
    try {
      const res = await axios.post("http://127.0.0.1:5000/predict", {
        message: message,
      });

      setResult(res.data.prediction);
      setModelAccuracy(res.data.model_accuracy);
    } catch (err) {
      console.error(err);
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-linear-to-br from-gray-900 to-gray-800 text-white flex items-center justify-center px-4">

      <div className="w-full max-w-2xl bg-gray-900 shadow-2xl rounded-2xl p-8 border border-gray-700">

        {/* Header */}
        <h1 className="text-3xl font-bold text-center mb-2">
          📩 Spam Detection System
        </h1>
        <p className="text-gray-400 text-center mb-6">
          Analyze messages using Machine Learning
        </p>

        {/* Input */}
        <textarea
          rows="5"
          placeholder="Enter your message here..."
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          className="w-full p-4 rounded-lg bg-gray-800 border border-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
        />

        {/* Button */}
        <button
          onClick={handleSubmit}
          className="mt-4 w-full bg-blue-600 hover:bg-blue-700 transition-all duration-200 py-3 rounded-lg font-semibold"
        >
          {loading ? "Analyzing..." : "Check Message"}
        </button>

        {/* Result Card */}
        {result && (
          <div className="mt-6 p-5 rounded-xl bg-gray-800 border border-gray-700 text-center">
            <h2 className="text-xl font-semibold mb-2">Result</h2>

            <span
              className={`px-4 py-2 rounded-full text-sm font-semibold ${
                result === "Spam"
                  ? "bg-red-500/20 text-red-400"
                  : "bg-green-500/20 text-green-400"
              }`}
            >
              {result}
            </span>

            {/* Accuracy */}
            {modelAccuracy !== null && (
              <p className="mt-3 text-gray-400">
                Model Accuracy:{" "}
                <span className="text-white font-medium">
                  {(modelAccuracy * 100).toFixed(2)}%
                </span>
              </p>
            )}
          </div>
        )}

      </div>
    </div>
  );
}

export default App;
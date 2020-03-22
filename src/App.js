import React, { useEffect, useState, useCallback } from "react";
import ml5 from "ml5";
import { csv } from "d3";
import "./App.css";

let nn; // the NeuralNetwork

function App() {
  const [nnReady, setNnReady] = useState(false);
  const [inputs, setInputs] = useState([]);
  const [results, setResults] = useState();

  const classify = useCallback(() => {
    if (!inputs || !inputs.length) return;
    // (!) classifyMultiple with all inputs (array of arrays) does not work
    // (!) classify (with single input array (inputs[0])) works
    nn.classifyMultiple(inputs[0], (error, results) => {
      if (error) {
        console.error(error);
        return;
      }
      setResults(results);
    });
  }, [inputs]);

  useEffect(() => {
    csv("./titanic/test.csv").then(testData => {
      setInputs(
        testData.map(passenger => [
          passenger.Sex,
          passenger.Age,
          passenger.Pclass,
          passenger.SibSp,
          passenger.Parch
        ])
      );
    });
  }, []);

  useEffect(() => {
    csv("./titanic/train.csv").then(trainingData => {
      nn = ml5.neuralNetwork({
        inputs: 5,
        outputs: 1,
        task: "classification",
        debug: false
      });

      trainingData.forEach(row => {
        const { Sex, Age, Pclass, SibSp, Parch, Survived } = row;
        nn.addData([Sex, Age, Pclass, SibSp, Parch], [Survived]);
      });

      nn.normalizeData();

      const trainingOptions = {
        epochs: 12, // what to choose here?
        batchSize: 12
      };

      nn.train(
        trainingOptions,
        epoch => {
          console.log("whileTraining", epoch);
          classify();
        },
        () => {
          console.log("finishedTraining");
          setNnReady(true);
          classify();
        }
      );
    });
  }, [classify]);

  return (
    <div className="App">
      <h1>ML5 Titanic</h1>
      <p>
        <small>
          {nnReady ? "NeuralNetwork ready!" : "Training NeuralNetwork..."}
        </small>
      </p>
      {results && (
        <>
          <h2>Results</h2>
          <pre>{JSON.stringify(results, null, 2)}</pre>
        </>
      )}
    </div>
  );
}

export default App;

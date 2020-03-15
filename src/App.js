import React, { useEffect, useState } from "react";
import ml5 from "ml5";
import { csv } from "d3";
import "./App.css";

let nn; // the NeuralNetwork

function App() {
  const [nnReady, setNnReady] = useState(false);
  const [testSubjects, setTestSubjects] = useState([]);
  const [selectedTestSubject, setSelectedTestSubject] = useState();
  const [results, setResults] = useState();

  const classify = id => {
    const passenger = testSubjects.find(
      passenger => passenger.PassengerId === id
    );

    if (!passenger) return;

    setSelectedTestSubject(passenger);

    const input = {
      Sex: passenger.Sex,
      Age: passenger.Age,
      Pclass: passenger.Pclass,
      SibSp: passenger.SibSp,
      Parch: passenger.Parch
    };

    nn.classify(input, (error, results) => {
      if (error) {
        console.error(error);
        return;
      }
      setResults(results);
    });
  };

  const handleResults = () => {
    const survivalChance = results.find(r => r.label === "1").confidence;
    return (
      <p>
        <strong>{Math.round(survivalChance * 100)}%</strong> sure that this
        passenger survived.
      </p>
    );
  };

  useEffect(() => {
    csv("./titanic/test.csv").then(testData => {
      setTestSubjects(testData);
    });
  }, []);

  useEffect(() => {
    csv("./titanic/train.csv").then(trainingData => {
      nn = ml5.neuralNetwork({
        inputs: ["Sex, Age, Pclass, SibSp, Parch"],
        outputs: ["Survived"],
        task: "classification",
        debug: true
      });

      trainingData.forEach(row => {
        const { Sex, Age, Pclass, SibSp, Parch, Survived } = row;
        nn.addData({ Sex, Age, Pclass, SibSp, Parch }, { Survived });
      });

      // nn.normalizeData();

      const trainingOptions = {
        epochs: 12, // what to choose here?
        batchSize: 12
      };

      nn.train(trainingOptions, () => {
        setNnReady(true);
      });
    });
  }, []);

  return (
    <div className="App">
      <h1>ML5 Titanic</h1>
      <p>
        <small>
          {nnReady ? "NeuralNetwork ready!" : "Training NeuralNetwork..."}
        </small>
      </p>

      <h2>Would this passenger survive?</h2>
      <select defaultValue="default" onChange={e => classify(e.target.value)}>
        <option key="default">-- Select test subject --</option>
        {testSubjects.map(sample => (
          <option key={sample.PassengerId} value={sample.PassengerId}>
            {sample.Name}
          </option>
        ))}
      </select>
      {selectedTestSubject && (
        <pre>{JSON.stringify(selectedTestSubject, null, 2)}</pre>
      )}
      {results && (
        <>
          <h2>Result</h2>
          {handleResults(results)}
          <pre>{JSON.stringify(results, null, 2)}</pre>
        </>
      )}
    </div>
  );
}

export default App;

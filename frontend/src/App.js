import React, { useState, useEffect } from 'react';
import './App.css';
import CardList from './components/CardList';
import SelectedDeck from './components/SelectedDeck';
import PredictedCards from './components/PredictedCards';

function App() {
  const [cards, setCards] = useState([]);
  const [cardMap, setCardMap] = useState(new Map());
  const [selectedCards, setSelectedCards] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Fetch card data and create mappings
    const fetchData = async () => {
      try {
        const cardsResponse = await fetch('./cards.json');
        const cardsData = await cardsResponse.json();
        setCards(cardsData);

        const csvResponse = await fetch('./number_to_card.csv');
        const csvText = await csvResponse.text();
        const cardNames = csvText.split('\n')[1].split(',');
        const newCardMap = new Map();
        cardNames.forEach((name, index) => {
          const card = cardsData.find(c => c.name === name);
          if (card) {
            newCardMap.set(index, card);
          }
        });
        setCardMap(newCardMap);
      } catch (err) {
        setError('Failed to load card data.');
      }
    };
    fetchData();
  }, []);

  const handleCardSelect = (card) => {
    if (selectedCards.length < 8 && !selectedCards.find(c => c.id === card.id)) {
      setSelectedCards([...selectedCards, card]);
    }
  };

  const handleCardRemove = (cardToRemove) => {
    setSelectedCards(selectedCards.filter(card => card.id !== cardToRemove.id));
  };

  const handleGeneratePredictions = async () => {
    if (selectedCards.length === 0) return;
    setLoading(true);
    setError(null);
    setPredictions([]);

    const selectedCardIds = selectedCards.map(card => {
        for (const [key, value] of cardMap.entries()) {
            if (value.id === card.id) {
                return key;
            }
        }
        return -1; // Should not happen with valid data
    }).filter(id => id !== -1);
    console.log(selectedCardIds)

    try {
        const MAX_LENGTH = window.APP_CONFIG?.MAX_LENGTH || 7;
        const PADDING_IDX = window.APP_CONFIG?.PADDING_IDX || 120;
        const INFERENCE_URL = window.APP_CONFIG?.INFERENCE_URL || "http://clashroyale.ddns.net/v2/models/clashroyale/infer";


        let paddedList = [...selectedCardIds];
        if (paddedList.length < MAX_LENGTH) {
            const numPadding = MAX_LENGTH - paddedList.length;
            paddedList = Array(numPadding).fill(PADDING_IDX).concat(paddedList);
        } else if (paddedList.length > MAX_LENGTH) {
            paddedList = paddedList.slice(paddedList.length - MAX_LENGTH);
        }


        const requestBody = {
            "inputs": [
                {
                    "name": "input-0",
                    "shape": [1, MAX_LENGTH],
                    "datatype": "INT64",
                    "data": [paddedList]
                }
            ]
        };

        const response = await fetch(INFERENCE_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const responseData = await response.json();
        const outputData = responseData.outputs[0].data;

        // Simple softmax
        const exps = outputData.map(Math.exp);
        const sumExps = exps.reduce((a, b) => a + b, 0);
        const probabilities = exps.map(e => e / sumExps);

        const top5 = probabilities
            .map((prob, index) => ({ prob, index }))
            .sort((a, b) => b.prob - a.prob)
            .slice(0, 5);

        const predictedCards = top5.map(p => ({
            card: cardMap.get(p.index),
            probability: p.prob
        }));

        setPredictions(predictedCards);
    } catch (err) {
        setError('Failed to get predictions from the model.');
        console.error(err);
    } finally {
        setLoading(false);
    }
  };


  return (
    <div className="App">
      <header className="App-header">
        <h1>Clash Royale Deck Creator</h1>
      </header>
      <main>
        <div className="deck-builder">
            <SelectedDeck
                selectedCards={selectedCards}
                onCardRemove={handleCardRemove}
                onGenerate={handleGeneratePredictions}
                />
            {loading && <p>Loading predictions...</p>}
            {error && <p className="error">{error}</p>}
            {predictions.length > 0 && <PredictedCards predictedCards={predictions} onCardSelect={handleCardSelect} />}
        </div>
        <CardList cards={cards} onCardSelect={handleCardSelect} />
      </main>
    </div>
  );
}

export default App;
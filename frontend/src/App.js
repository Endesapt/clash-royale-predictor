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
    // This useEffect hook for fetching data remains the same...
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

  // New function to clear the deck
  const handleClearDeck = () => {
    setSelectedCards([]);
    setPredictions([]);
    setError(null);
  };
  const getPredictionsFromModel = async (currentDeck) => {
    const selectedCardIds = currentDeck.map(card => {
        for (const [key, value] of cardMap.entries()) {
            if (value.id === card.id) return key;
        }
        return -1;
    }).filter(id => id !== -1);

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
        inputs: [{ name: "input-0", shape: [1, MAX_LENGTH], datatype: "INT64", data: [paddedList] }]
    };

    const response = await fetch(INFERENCE_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
    });

    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

    const responseData = await response.json();
    const outputData = responseData.outputs[0].data;
    const exps = outputData.map(Math.exp);
    const sumExps = exps.reduce((a, b) => a + b, 0);
    const probabilities = exps.map(e => e / sumExps);

    // --- FIX IS HERE ---

    // 1. Create an array of objects, pairing each probability with its original index
    const allPredictions = probabilities.map((prob, index) => ({
      index: index, // The actual card index predicted by the model
      probability: prob
    }));

    // 2. Sort this array to find the predictions with the highest probability
    allPredictions.sort((a, b) => b.probability - a.probability);

    // 3. Now, map the top-rated predictions to their actual card data
    const topPredictions = allPredictions
      .slice(0, 15) // Get a slightly larger slice to have options after filtering
      .map(p => ({
        card: cardMap.get(p.index), // Use the correct index from the sorted list
        probability: p.probability
      }))
      .filter(p => p.card); // Ensure the card exists in our map

    // --- END OF FIX ---

    // Filter out cards already in the deck
    const currentDeckIds = new Set(currentDeck.map(c => c.id));
    return topPredictions.filter(p => !currentDeckIds.has(p.card.id));
  };

  // Renamed from handleGeneratePredictions to be more specific
  const handleSuggestNextCards = async () => {
    if (selectedCards.length === 0 || selectedCards.length >= 8) return;
    setLoading(true);
    setError(null);
    setPredictions([]);

    try {
        const filteredPredictions = await getPredictionsFromModel(selectedCards);
        setPredictions(filteredPredictions.slice(0, 5)); // Show top 5 valid predictions
    } catch (err) {
        setError('Failed to get suggestions from the model.');
        console.error(err);
    } finally {
        setLoading(false);
    }
  };

  // New function to generate a full deck automatically
  const handleGenerateFullDeck = async () => {
    if (selectedCards.length === 0 || selectedCards.length >= 8) return;
    setLoading(true);
    setError(null);
    setPredictions([]);

    let currentDeck = [...selectedCards];

    try {
      while (currentDeck.length < 8) {
        const filteredPredictions = await getPredictionsFromModel(currentDeck);

        if (filteredPredictions.length === 0) {
          throw new Error("Model could not suggest a valid next card.");
        }

        // Choose one of the top 2 valid predictions
        const poolSize = Math.min(2, filteredPredictions.length);
        const chosenPrediction = filteredPredictions[Math.floor(Math.random() * poolSize)];
        
        currentDeck.push(chosenPrediction.card);
      }
      setSelectedCards(currentDeck);
    } catch (err) {
        setError(`Failed to generate a full deck. ${err.message}`);
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
                onSuggest={handleSuggestNextCards}
                onGenerateFullDeck={handleGenerateFullDeck}
                onClear={handleClearDeck}
                />
            {loading && <p>Loading...</p>}
            {error && <p className="error">{error}</p>}
            {predictions.length > 0 && <PredictedCards predictedCards={predictions} onCardSelect={handleCardSelect} />}
        </div>
        <CardList cards={cards} onCardSelect={handleCardSelect} />
      </main>
    </div>
  );
}

export default App;
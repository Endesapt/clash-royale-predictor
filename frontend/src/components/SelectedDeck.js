import React from 'react';

function SelectedDeck({ selectedCards, onCardRemove, onSuggest, onGenerateFullDeck, onClear }) {
    const isDeckFull = selectedCards.length >= 8;
    const isDeckEmpty = selectedCards.length === 0;

    return (
        <div className="selected-deck-container">
            <div className="deck-header">
                <h2>Your Deck ({selectedCards.length}/8)</h2>
                <button 
                    onClick={onClear} 
                    disabled={isDeckEmpty}
                    className="clear-button"
                >
                    Clear
                </button>
            </div>
            <div className="selected-cards">
                {selectedCards.map(card => (
                    <div key={card.id} className="selected-card" onClick={() => onCardRemove(card)}>
                        <img src={card.iconUrls.medium} alt={card.name} />
                    </div>
                ))}
            </div>
            <div className="deck-controls">
                <div className="control-item">
                    <button 
                        onClick={onGenerateFullDeck} 
                        disabled={isDeckFull || isDeckEmpty}
                    >
                        Generate Full Deck
                    </button>
                </div>
                <div className="control-item">
                    <button 
                        onClick={onSuggest} 
                        disabled={isDeckFull || isDeckEmpty}
                    >
                        Suggest Next Cards
                    </button>
                    <p className="explanation">
                        This suggests the next best cards but does NOT generate a whole deck for you.
                    </p>
                </div>
            </div>
        </div>
    );
}

export default SelectedDeck;
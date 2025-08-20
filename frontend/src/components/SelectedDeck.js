import React from 'react';

function SelectedDeck({
    selectedCards,
    onCardRemove,
    onGenerate
}) {
    return ( <div className = "selected-deck-container" >
        <h2 > Your Deck </h2> <div className = "selected-cards" > {
            selectedCards.map(card => ( <div key = {
                    card.id
                }
                className = "selected-card"
                onClick = {
                    () => onCardRemove(card)
                } >
                <img src = {
                    card.iconUrls.medium
                }
                alt = {
                    card.name
                }
                /> </div>
            ))
        } </div> <button onClick = {
            onGenerate
        }
        disabled = {
            selectedCards.length === 0
        } >
        Generate Next Cards </button> </div>
    );
}

export default SelectedDeck;
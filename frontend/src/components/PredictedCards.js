import React from 'react';
import Card from './Card';

function PredictedCards({
    predictedCards,
    onCardSelect
}) {
    return ( <div className = "predicted-cards-container" >
        <h2 > Predicted Cards </h2> <div className = "predicted-cards" > {
            predictedCards.map(({
                card,
                probability
            }) => ( <div key = {
                    card.id
                }
                className = "prediction" >
                <
                Card card = {
                    card
                }
                onCardClick = {
                    onCardSelect
                }
                /> <p > {
                    (probability * 100).toFixed(2)
                } % </p> </div>
            ))
        } </div> </div>
    );
}

export default PredictedCards;
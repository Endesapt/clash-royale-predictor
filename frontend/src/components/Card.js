import React from 'react';

function Card({
    card,
    onCardClick
}) {
    return ( <div className = "card"
        onClick = {
            () => onCardClick(card)
        } >
        <img src = {
            card.iconUrls.medium
        }
        alt = {
            card.name
        }
        /> <p > {
            card.name
        } </p> </div>
    );
}

export default Card;
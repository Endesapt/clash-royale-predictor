import React,
{
    useState
}
    from 'react';
import Card from './Card';

function CardList({
    cards,
    onCardSelect
}) {
    const [searchTerm, setSearchTerm] = useState('');

    const filteredCards = cards.filter(card =>
        card.name.toLowerCase().includes(searchTerm.toLowerCase())
    );

    return (
        <div className="card-list-container" >
        <input type="text"
            placeholder="Search for a card..."
            className="search-bar"
            onChange={
                e => setSearchTerm(e.target.value)
            }
        /> <div className="card-list" > {
                filteredCards.map(card => (<Card key={
                        card.id
                    }
                    card={
                        card
                    }
                    onCardClick={
                        onCardSelect
                    }
                />
                ))
            } </div> </div>
            );
}

            export default CardList;
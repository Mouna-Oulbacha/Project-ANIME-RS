// Function to open the modal and populate data
function openModal(title, genre, type, episodes, members, rating, image) {
  document.getElementById("animeTitle").textContent = title;
  document.getElementById("animeGenre").textContent = genre;
  document.getElementById("animeType").textContent = type;
  document.getElementById("animeEpisodes").textContent = episodes;
  document.getElementById("animeMembers").textContent = members;
  document.getElementById("animeRating").textContent = rating;
  document.getElementById("animeImage").src = image;
  document.getElementById("animeModal").style.display = "flex";

  // Render stars
  renderStars();
}

// Function to close the modal
function closeModal() {
  document.getElementById("animeModal").style.display = "none";
}

// Render star rating system dynamically
function renderStars() {
  const starContainer = document.getElementById("starRating");
  starContainer.innerHTML = ""; // Clear any existing stars

  for (let i = 1; i <= 10; i++) {
    const star = document.createElement("span");
    star.innerHTML = "★";
    star.dataset.value = i;
    star.addEventListener("click", setRating);
    starContainer.appendChild(star);
  }
}

// Highlight and store the selected rating
function setRating(event) {
  const value = parseInt(event.target.dataset.value);
  const stars = document.querySelectorAll("#starRating span");

  stars.forEach((star, index) => {
    if (index < value) {
      star.classList.add("active");
    } else {
      star.classList.remove("active");
    }
  });

  // Store the rating value
  document.getElementById("userRating").value = value;
}

// Submit rating
function submitRating() {
  const rating = document.getElementById("userRating").value;
  if (rating > 0) {
    alert(`You rated this anime ${rating}/10!`);
    closeModal();
  } else {
    alert("Please select a rating before submitting.");
  }
}

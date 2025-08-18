// /static/app.js

let game_number = 1;

async function makeDecision() {
  const day = document.getElementById('day').innerText;
  const demand = document.getElementById('demand').innerText;
  const userReplenishment = document.getElementById('replenishment').value;
  const agentState = document.getElementById('agentState').innerText; 
  const userState = document.getElementById('userState').innerText; 

  // Send decision to backend
  const response = await fetch('/api/make-decision', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      day: parseInt(day),  // Ensure day is sent as a number
      userReplenishment: parseInt(userReplenishment),  // Convert to integer
      currentAgentState: agentState.replace('[', '').replace(']', '').split(',').map(Number),
      currentUserState: userState.replace('[', '').replace(']', '').split(',').map(Number)
    }),
  });

  const result = await response.json();
  console.log(result);

  // Update frontend with the new state
  document.getElementById('day').innerText = result.day;  // Update day
  document.getElementById('demand').innerText = parseInt(result.demand);  // Update day
  document.getElementById('agentCost').innerText = result.agentCost.toFixed(2);  // Update agent cost
  document.getElementById('agentReplenishment').innerText = parseInt(result.agentReplenishment);
  document.getElementById('userCost').innerText = result.userCost.toFixed(2);  // Update user cost
  document.getElementById('totalAgentCost').innerText = result.totalAgentCost.toFixed(2);  // Update total agent cost
  document.getElementById('totalUserCost').innerText = result.totalUserCost.toFixed(2);  // Update total user cost
  document.getElementById('agentState').innerText = JSON.stringify(result.newAgentState);  // Update agent state
  document.getElementById('userState').innerText = JSON.stringify(result.newUserState);  // Update user state

  // Add new row to the table for the current day
  const tableBody = document.querySelector('#gameRecords tbody');
  const newRow = document.createElement('tr');
  newRow.innerHTML = `
    <td>${parseInt(game_number)}</td>
    <td>${parseInt(result.indexDay)}</td>
    <td>${userReplenishment}</td>
    <td>${parseInt(result.agentReplenishment)}</td>
    <td>${parseInt(result.demand)}</td>
    <td>${parseFloat(result.fulfillmentMetric)}</td>
    <td>${result.chosenInventory}</td>
    <td>${result.userCost.toFixed(2)}</td>
    <td>${result.agentCost.toFixed(2)}</td>
    <td>${JSON.stringify(userState.replace('[', '').replace(']', '').split(',').map(Number))}</td>
    <td>${JSON.stringify(agentState.replace('[', '').replace(']', '').split(',').map(Number))}</td>
    <td>${JSON.stringify(result.newUserState)}</td>
    <td>${JSON.stringify(result.newAgentState)}</td>
  `;
  tableBody.appendChild(newRow);  // Add the new row to the table

  // Scroll the table to the bottom to ensure new record is visible
  tableBody.scrollIntoView(false);


  // Check if the game is done
  if (result.done) {
    const gameOutcomeDiv = document.getElementById('gameOutcome');
    let message = '';

    if (result.totalAgentCost < result.totalUserCost) {
        message = '<h1 style="color: red; font-size: 30px; font-weight: bold;">Game Over! Agent Wins!</h1>';
    } else if (result.totalAgentCost > result.totalUserCost){
        message = '<h1 style="color: green; font-size: 30px; font-weight: bold;">Game Over! You Win!</h1>';
    } else {
        message = '<h1 style="color: yellow; font-size: 30px; font-weight: bold;">Game Over! Tie</h1>';
    }

    message += '<p style="font-size: 20px; font-style: italic;">To play a new game, reset the game!</p>';

    gameOutcomeDiv.innerHTML = message;
    gameOutcomeDiv.style.display = 'block';
  }

}

async function resetGame() {
  const response = await fetch('/api/reset-game', {
      method: 'POST'
  });

  const result = await response.json();

  if (result.status === 'Game reset successful') {
      // Reset UI elements
      game_number++;
      document.getElementById('game_number').innerText = parseInt(game_number);
      document.getElementById('day').innerText = 1;
      document.getElementById('agentCost').innerText = 0
      document.getElementById('userCost').innerText = 0
      document.getElementById('totalAgentCost').innerText = 0
      document.getElementById('totalUserCost').innerText = 0
      document.getElementById('agentState').innerText = JSON.stringify([0,0,0])
      document.getElementById('userState').innerText = JSON.stringify([0,0,0])

      document.getElementById('gameOutcome').innerText = "";
  }
}
function resetRecordTable() {
  const tableBody = document.querySelector('#gameRecords tbody');
  tableBody.innerHTML = ''; // Clear the table body
}

// Function to open the modal
function openDocumentation() {
  document.getElementById('docModal').style.display = 'block';
}

// Function to close the modal
function closeDocumentation() {
  document.getElementById('docModal').style.display = 'none';
}

// Close the modal if the user clicks outside of it
window.onclick = function(event) {
  const modal = document.getElementById('docModal');
  if (event.target === modal) {
    modal.style.display = 'none';
  }
};
// Toggle the hyperparameters popup
function toggleHyperparametersPopup() {
  const popup = document.getElementById('hyperparametersPopup');
  if (popup.style.display === 'none' || popup.style.display === '') {
    popup.style.display = 'block';
  } else {
    popup.style.display = 'none';
  }
}
// Attach event listener to the hyperparameters button
document.getElementById('hyperparametersBtn').addEventListener('click', toggleHyperparametersPopup);

function exportToExcel() {
  // Get the table element
  const table = document.getElementById('gameRecords');

  // Create a new workbook and a new worksheet
  const wb = XLSX.utils.book_new();
  const ws = XLSX.utils.table_to_sheet(table);

  // Append the worksheet to the workbook
  XLSX.utils.book_append_sheet(wb, ws, 'Records');

  // Generate a download link and click it
  XLSX.writeFile(wb, 'game_records.xlsx');
}


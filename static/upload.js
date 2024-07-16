// Display notification
function showNotification(text, type = "check") {
    const iconElement = document.getElementById('icon-message');
    const notificationElement = document.getElementById('notification');

    switch (type) {
        case 'check':
            iconElement.className = 'fa fa-check-circle';
            d3.select('#notification').classed("warning", false);
            d3.select('#notification').classed("alert", false);
            break;
        case 'warning':
            iconElement.className = 'fa fa-exclamation-circle';
            d3.select('#notification').classed("warning", true);
            break;
        case 'alert':
            iconElement.className = 'fa fa-cogs';
            d3.select('#notification').classed("alert", true);
            break;
    }

    d3.select("#text-not").text(text);
    d3.select("#notification").classed("show", true);
    setTimeout(function() {
        d3.select("#notification").classed("show", false);
    }, 8000);
}

// Get the file name and display
document.getElementById('csv').addEventListener('change', function() {
    var fileName = this.files[0].name;
    document.getElementById('file-name').textContent = fileName;
});

// Get the inicial selection of sex
const selectedIcon = document.querySelector('#sex-icons i.selected');
document.getElementById('sex').value = selectedIcon.getAttribute('data-sex');
// Get when change the selection
document.querySelectorAll('#sex-icons i').forEach(icon => {
    icon.addEventListener('click', function() {
        document.querySelectorAll('#sex-icons i').forEach(i => i.classList.remove('selected'));
        this.classList.add('selected');
        document.getElementById('sex').value = this.getAttribute('data-sex');
    });
});

// Add event listener to the button upload
document.getElementById('upload-button').addEventListener('click', () => {
    d3.select('.modal').classed("show", true);
    d3.select('#form-container').style('display', 'flex');
});

// Function to send info and upload data
const form = document.getElementById('ecgForm');
document.querySelector('#upload-info').addEventListener('click', function(event) {
    event.preventDefault(); // To avoid that refresh de page
    
    // Display loader and hide form
    d3.select('#loader').style('display', 'block');
    d3.select('#form-container').style('display', 'none');
    const formData = new FormData(form); // Get info form Form

    fetch(form.action, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(response => { })
    .catch((error) => {
        console.error('Error:', error);
    })
    .finally(() => {
        fetchDataAndInitialize(); // Re load data
        d3.select('#loader').style('display', 'none');
        d3.selectAll('.modal').classed("show", false);
        showNotification('Yay! 🎉 Your data has been uploaded successfully!');
    });
});

// Function to update selected for newdots
function update_nearby(id, selected, selected_id){
    fetch('/update_nearby', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json'},
        body: JSON.stringify({
            id: id,
            selected: selected,
            selected_id: selected_id
        }),
    })
    .then(response => response.json())
    .then(response => {
        d3.selectAll("#nearby3 .ecg").remove();
        d3.selectAll('.send-to-container').classed("show", false);
    })
    .catch((error) => {console.log('Error:', error);})
    .finally(() => {
        fetchDataAndInitialize(); // Re load data
        showNotification('Awesome! ✨ Your record has been updated successfully!');
    });

}



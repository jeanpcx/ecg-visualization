// Get the file name and display
document.getElementById('csv').addEventListener('change', function() {
    var fileName = this.files[0].name;

    var fileUploadElement = document.getElementById('file-upload');
    fileUploadElement.textContent = fileName;
    fileUploadElement.classList.add('check');   

});

function resetUploadLabel(){
    var fileUploadElement = document.getElementById('file-upload');
    fileUploadElement.textContent = "Choose your file";
    fileUploadElement.classList.remove('check');   
}

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
    // d3.select('#form-container').style('display', 'flex');
});

// Function to send info and upload data
const form = document.getElementById('ecgForm');
document.querySelector('#upload-info').addEventListener('click', function(event) {
    event.preventDefault(); // To avoid that refresh de page
    
    // Display loader and hide form
    d3.select('#loader-container').style('display', 'block');
    d3.select('#form-container').style('display', 'none');
    const formData = new FormData(form); // Get info form Form

    fetch(form.action, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(response => { 
        d3.select('#loader-container').style('display', 'none');
        d3.select('#result-container').style('display', 'flex');

        d3.select("#p-prediction").text(response.label);
        d3.select("#result-prediction").style('--color', classColor[response.pred]);

        d3.select("#p-age").text(response.age);
        d3.select("#p-sex").text(response.sex);


        if (response.sex == "Men"){
            d3.select("#i-sex").attr("class", "fa fa-person");
        }else{
            d3.select("#i-sex").attr("class", "fa fa-person-dress");
        }

        console.log(response)
    })
    .catch((error) => {
        console.error('Error:', error);
    })
    .finally(() => {
        fetchDataAndInitialize(); // Re load data
        resetUploadLabel(); // Set original text for lable Upload
    });
});

// Function to update selected for newdots
function updateNearby(id, selected, selected_id){
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
        clearSignalBlock(3);
    })
    .catch((error) => {console.log('Error:', error);})
    .finally(() => {
        fetchDataAndInitialize(); // Re load data
        showNotification('Awesome! ✨ Your record has been updated successfully!');
    });

}


document.getElementById('go-explore').addEventListener('click', () => {
    d3.select('.modal').classed("show", false);
    d3.select('#result-container').style("display", "none");
    d3.select('#form-container').style("display", "flex");
    // d3.select('#form-container').style('display', 'flex');
    showNotification('Yay! 🎉 Your data has been uploaded successfully!');
});
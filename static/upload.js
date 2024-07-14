document.getElementById('csv').addEventListener('change', function() {
    var fileName = this.files[0].name;
    document.getElementById('file-name').textContent = fileName;
});


const selectedIcon = document.querySelector('#sex-icons i.selected');
document.getElementById('sex').value = selectedIcon.getAttribute('data-sex');

const fileInput = document.getElementById('csv');
const fileNameSpan = document.getElementById('file-name');
fileInput.addEventListener('change', function () {
    fileNameSpan.textContent = this.files[0].name;
});

document.querySelectorAll('#sex-icons i').forEach(icon => {
    icon.addEventListener('click', function() {
        document.querySelectorAll('#sex-icons i').forEach(i => i.classList.remove('selected'));
        this.classList.add('selected');
        document.getElementById('sex').value = this.getAttribute('data-sex');
    });
});


// // Add event listener to the button
document.getElementById('upload-button').addEventListener('click', () => {
    d3.select('.modal').style("display", "flex");
});
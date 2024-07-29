// Define colors
const classLabel = ['NORM', 'MI', 'STTC', 'CD', 'HYP'];
const classColor = ["gray", "blue", "darkorange", "green", "purple", "#03305A"];
const nearbyColor = ["#03305A", 'darkred', 'lightsalmon', 'orangered'];

// Initialize Zooms
var currentZoomScatter = d3.zoomIdentity;
var currentZoomSignal = d3.zoomIdentity;

// Initialize Scatter Plot
const scatterZone = document.getElementById('scatter-zone');
var scatterRect = scatterZone.getBoundingClientRect();

    // Define global dimensions and safe zone for Scatter Plot
var scatterWidth = scatterZone.clientWidth;
var scatterHeight = scatterZone.clientHeight;
const scatterMargin = { top: 20, right: 20, bottom: 20, left: 20 };
var graphWidth = scatterWidth - scatterMargin.left - scatterMargin.right;
var graphHeight = scatterHeight - scatterMargin.top - scatterMargin.bottom;

    // Define axes x and y
const scatterX = d3.scaleLinear().range([0, graphWidth]);
const scatterY = d3.scaleLinear().range([graphHeight, 0]);

// Initialize Signal dimensions
const signalMargin = { top: 10, right: 15, bottom: 25, left: 40 };
let signalX = d3.scaleLinear().domain([0, 1000]);
let signalY = d3.scaleLinear().domain([-1, 1]);

// Initialize minimap dimensions
const mapWidth = 80;
const mapHeight = 80;
const mapSVG = d3.select("#minimap")
                .attr("width", mapWidth)
                .attr("height", mapHeight);
const mapX = d3.scaleLinear().range([0, mapWidth]);
const mapY = d3.scaleLinear().range([mapHeight, 0]);
const mapGroup = mapSVG.append("g");
const mapContainer = d3.select("#minimap-container")
                .style("left", scatterRect.left + "px")
                .style("top", scatterRect.top + scatterRect.height - mapHeight+ "px");


var isMiniMapInitialized = false
function showMap(data, newData) {
    const dots = mapGroup.selectAll(".fix").data(data, d => d._id);
    dots.exit().remove();
    dots.attr("cx", d => mapX(d.x))
        .attr("cy", d => mapY(d.y))
        .style("fill", d => classColor[d.true_]);

    // Add new points
    dots.enter().append("circle")
        .attr("class", "dot fix")
        .attr("r", 0.6)
        .attr("cx", d => mapX(d.x))
        .attr("cy", d => mapY(d.y))
        .style("fill", d => classColor[d.true_])

    // Add dots to the minimap
    mapGroup.selectAll(".new-dot")
        .data(newData)
        .enter().append("circle")
        .attr("class", "dot new-dot")
        .attr("r", 1)
        .attr("cx", d => mapX(d.x))
        .attr("cy", d => mapY(d.y))
        .style("fill", "red");

    if (isMiniMapInitialized) return;

    // Create a rectangle to show the current zoom area
    const viewRect = mapGroup.append("rect")
        .attr("class", "viewRect")
        .attr("width", mapWidth)
        .attr("height", mapHeight)
        .style("fill", "none")
        .style("stroke", "red")
        .style("z-index", 1000);

    isMiniMapInitialized = true
}

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

// Initialize ToolTip
const tooltip = d3.select('.tooltip');

    // Define function to show or hide
function showTooltip(d, state = true){
    if (state) {
        // If is a new signal upload, only have prediction
        if (d.true_ === null) {
            var label = "Pred: " + d.pred_label;
            var color = classColor[d.pred];
        } else {
            // If signal have a true Dx
            var label = d.true_label;    
            var color = classColor[d.true_];
        }

        tooltip.classed("show", true)
            .html(d._id + "<br/><strong style='color:" + color + "'>" + label + "</strong><br/>" + d.age + " years <br/>" + d.sex)
            .style("left", scatterRect.left + "px")
            .style("top", scatterRect.top + "px")
            .style("--color", color);
    }else{
        tooltip.classed("show", false);
    }
}

// Define zoom behavior for Scatter Plot
var scatterZoom = d3.zoom()
    .scaleExtent([1, 30])
    .translateExtent([[0, 0], [scatterWidth, scatterHeight]])
    .on("zoom", function(event) {
        currentZoomScatter = event.transform;
        applyZoomScatter(currentZoomScatter);
        applyZoomMap(currentZoomScatter);
    });
    
    // Define zoom transform for Scatter Plot
function applyZoomScatter(transform) {
    let x = transform.rescaleX(scatterX);
    let y = transform.rescaleY(scatterY);

    d3.select("#scatter-plot .x.axis").call(d3.axisBottom(x));
    d3.select("#scatter-plot .y.axis").call(d3.axisLeft(y));
    d3.selectAll("#scatter-plot .dot").attr("cx", d => x(d.x)).attr("cy", d => y(d.y));
}

// Define zoom behavior for MiniMap
function applyZoomMap(transform) {
    const scaleX = mapWidth / (graphWidth * transform.k);
    const scaleY = mapHeight / (graphHeight * transform.k);
    const translateX = -transform.x * scaleX;
    const translateY = -transform.y * scaleY;

    d3.select('.viewRect').attr("x", translateX).attr("y", translateY)
        .attr("width", mapWidth / transform.k)
        .attr("height", mapHeight / transform.k);
}

// Initialize SVG container for scatter plot
const scatterSVG = d3.select("#scatter-plot")
    .append("svg").attr("id", "plotdot")
    .attr("width", scatterWidth).attr("height", scatterHeight)
    .call(scatterZoom)
    .append("g").attr("transform", "translate(" + scatterMargin.left + "," + scatterMargin.top + ")");

scatterSVG.append("rect")
    .attr("width", scatterWidth).attr("height", scatterHeight)
    .style("fill", "none")
    .style("pointer-events", "all")
    .on("click", function () {clearECGDisplays();});

// Define function to clear ECG displays
function clearECGDisplays() {

    clearSignalBlock(0);
    clearSignalBlock(1);
    clearSignalBlock(2);
    clearSignalBlock(3);



    // d3.selectAll(".ecg").remove();
    d3.selectAll(".dot.selected").classed("selected", false);
    d3.selectAll(".dot").classed("nearby1", false);
    d3.selectAll(".dot").classed("nearby2", false);
    d3.selectAll(".dot").classed("find-nearby1", false);
    d3.selectAll(".dot").classed("find-nearby2", false);
    d3.selectAll(".dot").classed("examine", false);
    d3.selectAll('.send-to-container').classed("show", false);
}

// Define function to create age slider
function createSliders(min_age, max_age){
    var ageSlider = document.getElementById('age-slider');

    // Check if the slider is already initialized
    if (ageSlider.noUiSlider) return ageSlider;

    // Create slider
    noUiSlider.create(ageSlider, {
        start: [40, 70],
        connect: true,
        range: { 'min': min_age, 'max': 100},
        tooltips: true,
        format: {
            to: function (value) {return value.toFixed(0);},
            from: function (value) { return Number(value);}
        },
        pips: { mode: 'count', values: 6, density: 4}
    });

    return ageSlider // Return the d3 object
}

// Define function to create buttons
var isButtonsInitialized = false
function createButtons(classes){
    // Check if class butoms is already initialized
    if (isButtonsInitialized) return;    
    const buttonContainer = d3.select("#button-container");
    
    // Button for all, and set as selected
    buttonContainer.append("button")
        .attr("class", "button class-button selected")
        .attr("data-class", classes.length)
        .text("All")
        .style('--color', classColor[classes.length]);

    // Button for each class
    classes.forEach(clas => {
        buttonContainer.append("button")
            .attr("class", "button class-button selected")
            .attr("data-class", clas)
            .text(classLabel[clas])
            .style('--color', classColor[clas]);
    });
    isButtonsInitialized = true
}

// Define function to get parameters amd filter data
function getFilter(data){
    const sex = d3.select(".fa.selected").attr("data-sex");
    const ageRange = document.getElementById('age-slider').noUiSlider.get();
    let selectedClasses = d3.selectAll(".class-button.selected")
        .nodes().map(button => button.getAttribute("data-class"));

    // If no class is selected
    if (selectedClasses.length === 0) return [];

    // If all buttom is selected, remove '5'
    if (selectedClasses.includes("5")) {
        selectedClasses = selectedClasses.slice(1);
    }

    // Filter data
    const filtered_data = data.filter(d => {
        let clas = d.true_
        return (selectedClasses.includes(clas.toString())) &&
               (sex === "ALL" || d.sex === sex) &&
               (d.age >= ageRange[0] && d.age <= ageRange[1]);
    });

    return filtered_data
}

//  Define functions to plot a signal
function normalizeSignal(data) {
    const maxVal = d3.max(data);
    const minVal = d3.min(data);
    return data.map(d => 2 * (d - minVal) / (maxVal - minVal) - 1);
}

function smoothSignal(data, windowSize) {
    data = normalizeSignal(data);
    return data.map((entry, index, array) => {
        const start = Math.max(0, index - windowSize);
        const end = Math.min(array.length, index + windowSize + 1);
        const subset = array.slice(start, end);
        const average = subset.reduce((acc, val) => acc + val, 0) / subset.length;
        return average;
    });
}

function clearSignalBlock(id){
    const container = d3.select("#nearby" + id);
    container.select('.ecg-header').select('.ecg-info').selectAll('*').remove();
    container.select('.ecg').select(".ecg-plot").selectAll('*').remove();
    container.select('.ecg').style('--color', 'gray');
    if(id==3){
        d3.select("#ecg-inspect").classed('show', false);
    }
}

function plotSignal(d, signal, containerId = 0){
    const container = d3.select("#nearby" + containerId);

    // Get blocks for signal and info
    const signalZone = container.select('.ecg');
    const signalInfo = container.select('.ecg-header').select('.ecg-info');
    const signalPlot = signalZone.select(".ecg-plot");

    
    // Define color for margin, if is the selected point: same as class
    var marginColor = (containerId == 0) ? classColor[d.true_] : nearbyColor[containerId];

    // Check if is a new signal (Only prediction), display PRED:
    if (d.true_ === null) {
        signalInfo.append("div").attr("class", "info-block")
        .html(`<p style="color:white;">Pred: ${d.pred_label}</p>`)
        .style("background-color", classColor[d.pred])
        .style("color", "white");
        marginColor = classColor[d.pred];
    } else {
        signalInfo.append("div").attr("class", "info-block")
        .html(`<p style="color:white;">${d.true_label}</p>`)
        .style("background-color", classColor[d.true_])
        .style("color", "white");
    }
    signalInfo.append("div").attr("class", "info-block").html(`<p>${d.sex}</p>`);
    signalInfo.append("div").attr("class", "info-block").html(`<p>${d.age} years</p>`);
    signalZone.style("--color", marginColor);

    // Get local dimensions
    var signalWidth = signalPlot.node().clientWidth;
    var signalHeight = signalPlot.node().clientHeight;
    var graphWidthSignal = signalWidth - signalMargin.left - signalMargin.right;
    var graphHeightSignal = signalHeight - signalMargin.top - signalMargin.bottom;

    // Create SVG for ECG
    const signalSVG = signalPlot.append("svg")
        .attr("width", signalWidth)
        .attr("height", signalHeight)
        .append("g")
        .attr("transform", "translate(" + signalMargin.left + "," + signalMargin.top + ")");

    // Define clip path
    signalSVG.append("defs").append("clipPath")
        .attr("id", "clip-" + d._id)
        .append("rect")
        .attr("width", graphWidthSignal)
        .attr("height", signalHeight);

    const signalGroup = signalSVG.append("g").attr("clip-path", "url(#clip-" + d._id + ")");

    signalGroup.append("rect")
        .attr("width", graphWidthSignal)
        .attr("height", signalHeight)
        .style("fill", "none")
        .style("pointer-events", "all")

    if (containerId  >= 0){
        signalGroup.call(d3.zoom()
        .scaleExtent([1, 10])
        .translateExtent([[0, 0], [graphWidthSignal, graphHeightSignal]])
        .extent([[0, 0], [graphWidthSignal, graphHeightSignal]])
        .on("zoom", (event) => {
            currentZoomSignal = event.transform;
            applyZoomSignal(currentZoomSignal);
        }));
    }
        

    // Define axes for ECG
    signalX.range([0, graphWidthSignal]);
    signalY.range([graphHeightSignal, 0]);

    // Draw Signal: First lead and x axis
    const smoothedSignal = smoothSignal(signal, 5);
    signalGroup.append("path")
        .datum(smoothedSignal)
        .attr("class", "line")
        .attr("d", d3.line().x((d, i) => signalX(i)).y(d => signalY(d)));

        signalGroup.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + graphHeightSignal + ")")
        .call(d3.axisBottom(signalX));
    // Draw y axis
    signalSVG.append("g")
        .attr("class", "y axis")
        .call(d3.axisLeft(signalY));

    function applyZoomSignal(transform) {
        d3.selectAll(".ecg").each(function () {
            const svg = d3.select(this);
            // signalX.range([0, graphWidthSignal]);
            signalY.range([graphHeightSignal, 0]);

            const x = transform.rescaleX(signalX);
        
            svg.select(".x.axis").call(d3.axisBottom(x));
            svg.select(".line").attr("d", d3.line()
                .x((d, i) => x(i))
                .y(d => signalY(d))
            );
        });
    }
}

// Function to get signal from data, and graph
function getSignal(d, filter, renderSelected = true) {
    let filterIds = filter.map(d => d._id)
    fetch('/get_signals', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json'},
        body: JSON.stringify({
            id: d._id,
            filterIds: filterIds,
        }),
    })
        .then(response => response.json())
        .then(result => {
            const {nearbyIDs: nearbyIDs, selectedIds: selectedIds, signal: signals} = result;
            var ids = nearbyIDs

            if (renderSelected){
                clearSignalBlock(0);
                plotSignal(d, signals[0]);
            }

            if (signals.length > 1){
                d3.selectAll(".dot").classed("nearby1", false);
                d3.selectAll(".dot").classed("nearby2", false);
                d3.selectAll(".dot").classed("find-nearby1", false);
                d3.selectAll(".dot").classed("find-nearby2", false);
                
                d3.selectAll('.send-to-container').classed("show", false);

                // Show recommends points
                if (selectedIds != null){
                    d3.select("#dot-" + ids[1]).classed("find-nearby1", true); // Select new nearby
                    d3.select("#dot-" + ids[2]).classed("find-nearby2", true); // Select new nearby                  
                    ids = selectedIds
                }

                for (let i = 1; i < signals.length; i++) {
                    clearSignalBlock(i);
                    id = ids[i]
                    d3.select("#dot-" + id).classed("nearby" + i, true); // Select new nearby
                    d = filter.find(record => record._id === id);
                    
                    if (d){
                        plotSignal(d, signals[i], i);
                    }else{
                        showNotification('Oopsie! 😅 The nearest ' + i + ' point isnt within the filters', 'warning')
                    }
                }
            }
        })
        .catch((error) => {
            console.log('Error:', error);
        });
}

function examineSignal(d, content=3){
    fetch('/examine_signal', {
        // Send id of selected point to examine
        method: 'POST',
        headers: { 'Content-Type': 'application/json'},
        body: JSON.stringify({id: d._id}),
    })
        .then(response => response.json())
        .then(signal => {
            clearSignalBlock(content);
            if(content ==3){
                d3.selectAll('#ecg-inspect').classed("show", true);
            }
            plotSignal(d, signal, content);
        })
        .catch((error) => {
            console.log('Error:', error);
        });
}

function checkSelected(filteredData){
    // Check if any point is selected
    const selectedDot = d3.select(".dot.selected");
    if (selectedDot.empty()) {
        clearECGDisplays();
    } else {
        const element = selectedDot.node();
        const data = selectedDot.data()[0];
        
        // Check if selected point is still in filtered data or if selected point if new.
        if (element.classList.contains('new-dot') || filteredData.some(d => d._id === data._id)) {
            getSignal(data, filteredData, false);
        } else {
            showNotification('Oops! 🤔 Dot not found. Adjust your filter and try again.', 'warning');
            clearECGDisplays();
        }
    }
}

function update(data, newData) {
    var filteredData = getFilter(data);

     // Check if any cluster is selected
    if (filteredData.length === 0) {
        clearECGDisplays();
        scatterSVG.selectAll(".fix-dot").remove();
        mapSVG.selectAll(".fix").remove();
        return;
    }

    checkSelected(filteredData);
    showMap(filteredData, newData);

    const dots = scatterSVG.selectAll(".fix-dot").data(filteredData, d => d._id);
    dots.exit().remove();
    dots.attr("cx", d => scatterX(d.x))
        .attr("cy", d => scatterY(d.y))
        .style("fill", d => classColor[d.true_]);

    // Add new points
    dots.enter().append("circle")
        .attr("class", "dot fix-dot")
        .attr("r", 2.5)
        .attr("cx", d => scatterX(d.x))
        .attr("cy", d => scatterY(d.y))
        .attr("id", d => "dot-" + d._id)
        .style("fill", d => classColor[d.true_])
        .on("click", function (event, d) {
            const newDot = document.querySelectorAll('.new-dot.selected');
            
            if (newDot.length == 0) {
                // If no new point is selected: continue with the normal viewing process.
                d3.selectAll(".dot.selected").classed("selected", false);
                d3.select(this).classed("selected", true);
                let filteredData = getFilter(data);
                getSignal(d, filteredData);
            } else {
                // If new point is selected: The user can inspect other signals.
                d3.selectAll(".dot.examine").classed("examine", false);
                d3.select(this).classed("examine", true);
                examineSignal(d);
            }
        });

    // Add new points from data_new with gray color
    const newDots = scatterSVG.selectAll(".new-dot").data(newData, d => d._id);
    newDots.exit().remove();
    newDots.attr("cx", d => scatterX(d.x))
        .attr("cy", d => scatterY(d.y))
        .style("fill", d => classColor[d.pred]);

    newDots.enter().append("circle")
        .attr("class", "new-dot dot")
        .attr("r", 2.5)
        .attr("cx", d => scatterX(d.x))
        .attr("cy", d => scatterY(d.y))
        .attr("id", d => "new-dot-" + d._id)
        .style("fill", d => classColor[d.pred])
        .on("click", function (event, d) {
            d3.selectAll(".dot.selected").classed("selected", false);
            d3.selectAll(".dot.examine").classed("examine", false);

            d3.select(this).classed("selected", true);
            let filteredData = getFilter(data);

            if(filteredData.length > 0){
                getSignal(d, filteredData);
            }else{
                examineSignal(d, 0);
            }
        });

    scatterSVG.selectAll(".dot")
        .on("mouseover", function (event, d) {
            showTooltip(d); // Display Tooltip
        })
        .on("mouseout", function (event, d) {
            showTooltip(null, false); // Hide Tooltip
        });

    applyZoomScatter(currentZoomScatter);
    applyZoomMap(currentZoomScatter);

    d3.selectAll(".send-button").on("click", function(event) {
        // Nearby 1 or 2?
        var sendTo = d3.select(this).attr("send-to");
        const examineDot = d3.select(".dot.examine")
        const d = examineDot.data()[0]; // Info of examoine point
        const selectedDot = d3.select(".new-dot.selected").data()[0]; // Info of select point
    
        if (d && selectedDot) {
            selectedDot.selected_1 = d._id;
            const name = "nearby" + sendTo;
    
            d3.selectAll(".dot." + name).classed("find-" + name, true).classed(name, false)
            examineDot.classed(name, true);
            updateNearby(selectedDot._id, sendTo, d._id);
        } else {
        console.log("No hay un punto examine o new-dot seleccionado.");
        }
    });
}

// To load data when index is upload
document.addEventListener('DOMContentLoaded', function () {
    console.log("Initializing the application...")
    fetchDataAndInitialize();
});

function fetchDataAndInitialize() {
    console.log('Loading data')
    fetch('/get_data')
        .then(response => response.json())
        .then(result => {
            // Split the data with true labels, and new points with only prediction
            const data = result.filter(row => row.true_ !== null && row.true_ !== undefined && row.true_ !== '');
            const newData = result.filter(row => row.true_ === null || row.true_ === undefined || row.true_ === '');

            // Get range age and unique classes
            const minAge = d3.min(data, d => d.age);
            const maxAge = d3.max(data, d => d.age);
            const classes = [...new Set(data.map(d => d.true_))];

            // Create Button Classes and Age Slider
            const ageSlider = createSliders(minAge, maxAge);
            createButtons(classes);

            // Define limits for scatter Plot and Map
            scatterX.domain(d3.extent(data, d => d.x));
            scatterY.domain(d3.extent(data, d => d.y));
            mapX.domain(d3.extent(data, d => d.x));
            mapY.domain(d3.extent(data, d => d.y));

            // Initialice the Scatter Plot
            update(data, newData);

            // Update if any changes in button classes
            d3.selectAll(".class-button").on("click", function () {
                let btn = d3.select(this);
                // All selected
                if (this.getAttribute("data-class") === "5") {
                    if (btn.classed("selected")) {
                        // Deselect all buttons
                        d3.selectAll('.class-button').classed("selected", false);
                        showNotification('Hey there! 👋 Please pick at least one cluster to continue 😊', 'alert');
                    } else {
                        // Select all buttons
                        d3.selectAll('.class-button').classed("selected", true);
                    }
                } else {
                    // Change status of button
                    btn.classed("selected", !btn.classed("selected"));
                    const size = d3.selectAll(".class-button:not([data-class='5']).selected").size();
                    if (size < classes.length) {
                        // Deselect all button
                        d3.select('.class-button[data-class="5"]').classed("selected", false);
                    }
                    if (size === classes.length) {
                        // Select all button
                        d3.select('.class-button[data-class="5"]').classed("selected", true);
                    }
                    if (size == 0){
                        showNotification('Hey there! 👋 Please pick at least one cluster to continue 😊', 'alert');
                    }
                }
                update(data, newData);
            });

            // Update if any changes in selected sex
            d3.selectAll("#gender-icons i").on("click", function () {
                d3.selectAll("#gender-icons i").classed("selected", false);
                d3.select(this).classed("selected", true);
                update(data, newData);
            });

            // Update if any changes in age range
            ageSlider.noUiSlider.on('set', function (values, handle) {
                update(data, newData);
            });
        })
        .catch(error => {
            console.error('Error fetching data:', error);
        });
}
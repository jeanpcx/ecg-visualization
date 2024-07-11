window.addEventListener('resize', resizeSVGs);

// Define clusters names and colors
const clusterLabels = ["NORM", "MI", "STTC", "CD", "HYP"];
var cluster_color = ["gray", "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#03305A"]

// Define global dimensions and safe zones
const pageWidth = document.body.clientWidth;
const pageHeight = document.body.clientHeight;
const centerX = (pageWidth) * 0.25;
const centerY = (pageHeight) * 0.5;
const container = document.getElementById('scatter-zone');
var width = container.clientWidth;
var height = container.clientHeight;
const margin = { top: 20, right: 20, bottom: 20, left: 20 };
const margin_signal = { top: 10, right: 15, bottom: 25, left: 25 };
var graphWidth = width - margin.left - margin.right;
var graphHeight = height - margin.top - margin.bottom;

const tooltip = d3.select('.tooltip');

// Define Zoom behavior for scatter plot
let currentZoomState = d3.zoomIdentity;
let currentZoomECG = d3.zoomIdentity;
var zoom = d3.zoom()
    .scaleExtent([1, 30])
    .translateExtent([[0, 0], [width, height]])
    .on("zoom", function(event) {
        currentZoomState = event.transform;
        applyZoomTransform(event.transform);
    });

function applyZoomTransform(transform) {
    var new_x = transform.rescaleX(x);
    var new_y = transform.rescaleY(y);

    d3.select("#scatter-plot .x.axis").call(d3.axisBottom(new_x));
    d3.select("#scatter-plot .y.axis").call(d3.axisLeft(new_y));

    d3.selectAll("#scatter-plot .dot")
        .attr("cx", d => new_x(d.x))
        .attr("cy", d => new_y(d.y));
}

function zoomToPoint(d) {
    const pointX = x(d.x);
    const pointY = y(d.y);
    const scale = currentZoomState.k * 2;

    const translateX = width / 2 - pointX * scale;
    const translateY = height / 2 - pointY * scale;

    const transform = d3.zoomIdentity
        .translate(translateX, translateY)
        .scale(scale);

    svg.transition().duration(900).call(zoom.transform, transform);
}  

// Function to get color for hover buttons
function adjustSaturation(colorHex, targetSaturation) {
    let originalColor = chroma(colorHex);
    let adjustedColor = originalColor.set('hsv.s', targetSaturation);
    return adjustedColor.hex();
}

// Function to resize when window is resized
function resizeSVGs() {
    var newWidth = container.clientWidth;
    var newHeight = container.clientHeight;

    d3.select("#scatter-plot svg")
        .attr("width", newWidth)
        .attr("height", newHeight);

    x.range([0, newWidth - margin.left - margin.right]);
    y.range([newHeight - margin.top - margin.bottom, 0]);

    svg.selectAll(".dot")
        .attr("cx", d => x(d.x))
        .attr("cy", d => y(d.y));

    svg.call(zoom.transform, currentZoomState); // Reapply current zoom state after resize
}

// Create SVG container for scatter plot
const svg = d3.select("#scatter-plot")
    .append("svg")
    .attr("width", width)
    .attr("height", height)
    .call(zoom)
    .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

svg.append("rect")
    .attr("width", width)
    .attr("height", height)
    .style("fill", "none")
    .style("pointer-events", "all")
    .on("click", function () {
        d3.selectAll(".dot.selected").classed("selected", false);
        clearECGDisplays();
    });

// Define axes x and y
const x = d3.scaleLinear().range([0, graphWidth]);
const y = d3.scaleLinear().range([graphHeight, 0]);

// Function to clear ECG displays
function clearECGDisplays() {
    d3.selectAll(".ecg").remove();
    d3.selectAll(".ecg-content").classed("graph", false);
    d3.selectAll("#noneECG").style("display", 'block');
    d3.selectAll(".dot").classed("nearby", false);

}

// Function to create age slider
function create_sliders(min_age, max_age){
    var age_slider = document.getElementById('age-slider');
    noUiSlider.create(age_slider, {
        start: [40, 70],
        connect: true,
        range: {
            'min': 10, //min_age,
            'max': 90//max_age
        },
        tooltips: true,
        format: {
            to: function (value) {
                return value.toFixed(0); // No decimales
            },
            from: function (value) {
                return Number(value);
            }
        },
        pips: {
            mode: 'count',
            values: 6,
            density: 4
        }
    });
    return age_slider
}

// Function to create buttons
function create_buttons(clusters){
    const buttonContainer = d3.select("#button-container");

    // Button fo ALL
    buttonContainer.append("button")
        .attr("class", "cluster-button selected")
        .attr("data-cluster", clusters.length)
        .text("All")
        .style('--cluster-color', cluster_color[clusters.length]);

    // Button for each cluster
    clusters.forEach(cluster => {
        buttonContainer.append("button")
            .attr("class", "cluster-button selected")
            .attr("data-cluster", cluster)
            .text(clusterLabels[cluster])
            .style('--cluster-color', cluster_color[cluster]);
    });
}

// Function to get parameters amd filter data
function get_filter(data){
    var sex = d3.select(".fa.selected").attr("data-sex");
    var selectedClusters = d3.selectAll(".cluster-button.selected")
        .nodes()
        .map(button => button.getAttribute("data-cluster"));

    if (selectedClusters.length === 0) {
        return [];
    }

    if (selectedClusters.includes("5")) {
        selectedClusters = selectedClusters.slice(1);
    }

    var age_values = document.getElementById('age-slider').noUiSlider.get();
    var min = parseFloat(age_values[0]);
    var max = parseFloat(age_values[1]);

    let filtered_data = data.filter(d => {
        return (selectedClusters.includes(d.cluster.toString())) &&
               (sex === "ALL" || d.sex === sex) &&
               (d.age >= min && d.age <= max);
    });

    return filtered_data
}

// Function to manage tooltip state
function show_tooltip(event, d, state = true){
    if (state) {
        var tooltipX = event.pageX;
        var tooltipY = event.pageY;
        tooltip.classed("show", true)
            .html(d.id + "<br/><strong style='color:" + cluster_color[d.cluster] + "'>" + clusterLabels[d.cluster] + "</strong><br/>Age: " + d.age + " years <br/>Sex: " + (d.sex === "Hombre" ? "Men" : "Women"))
            .style("left", (tooltipX < centerX ? tooltipX + 10 : tooltipX - 95) + "px")
            .style("top", (tooltipY < centerY ? tooltipY + 10 : tooltipY - 60) + "px");
    }else{
        tooltip.classed("show", false);
    }
}

function normalize_signal(data) {
    const maxVal = d3.max(data);
    const minVal = d3.min(data);
    return data.map(d => 2 * (d - minVal) / (maxVal - minVal) - 1);
}

function smooth_signal(data, windowSize) {
    data = normalize_signal(data);
    return data.map((entry, index, array) => {
        const start = Math.max(0, index - windowSize);
        const end = Math.min(array.length, index + windowSize + 1);
        const subset = array.slice(start, end);
        const average = subset.reduce((acc, val) => acc + val, 0) / subset.length;
        return average;
    });
}

let x_signal = d3.scaleLinear().domain([0, 1000]);
let y_signal = d3.scaleLinear().domain([-1, 1]);

function draw_signal(d, signal, container_id){
    const container = d3.select(container_id);
    container.classed('graph', true);

    const signal_zone = container.append("div").attr("class", "ecg").attr("id", "ecg-" + d.id);
    const signal_info = signal_zone.append("div").attr("class", "ecg-info");
    const signal_plot = signal_zone.append("div").attr("class", "ecg-plot");

    signal_zone.on("mouseover", function () {
        d3.select("#dot-" + d.id).classed("highlight", true);
    }).on("mouseout", function () {
        d3.select("#dot-" + d.id).classed("highlight", false);
    });

    // Show MetaData
    // signal_info.append("div").attr("class", "info-block").html(`<p>${d.id}</p>`);
    signal_info.append("div").attr("class", "info-block")
        .html(`<p style="color:white;">${d.label}</p>`)
        .style("background-color", cluster_color[d.cluster])
        .style("color", "white");
    signal_info.append("div").attr("class", "info-block").html(`<p>${(d.sex === "Hombre" ? "Men" : "Women")}</p>`);
    signal_info.append("div").attr("class", "info-block").html(`<p>${d.age} years</p>`);

    // Get local dimensions
    const plot = document.getElementsByClassName('ecg-plot')[0];
    var width_signal = plot.clientWidth;
    var height_signal = 150;

    var graphWidth_signal = width_signal - margin_signal.left - margin_signal.right;
    var graphHeight_signal = height_signal - margin_signal.top - margin_signal.bottom;

    // Create SVG for ECG
    const svg = signal_plot.append("svg")
        .attr("width", width_signal)
        .attr("height", height_signal)
        .append("g")
        .attr("transform", "translate(" + margin_signal.left + "," + margin_signal.top + ")");

    // Define clip path
    svg.append("defs").append("clipPath")
        .attr("id", "clip-" + d.id)
        .append("rect")
        .attr("width", graphWidth_signal)
        .attr("height", height_signal);

    const g = svg.append("g").attr("clip-path", "url(#clip-" + d.id + ")");

    svg.append("rect")
        .attr("width", graphWidth_signal)
        .attr("height", height_signal)
        .style("fill", "none")
        .style("pointer-events", "all")
        .call(d3.zoom()
            .scaleExtent([1, 10])
            .translateExtent([[0, 0], [graphWidth_signal, graphHeight_signal]])
            .extent([[0, 0], [graphWidth_signal, graphHeight_signal]])
            .on("zoom", (event) => {
                signal_zoom(event.transform);
            }));

    // Define axes for ECG
    x_signal.range([0, graphWidth_signal]);
    y_signal.range([graphHeight_signal, 0]);

    // Draw Signal: First lead.
    const smoothed_signal = normalize_signal(signal, 5);
    g.append("path")
        .datum(smoothed_signal)
        .attr("class", "line")
        .attr("d", d3.line().x((d, i) => x_signal(i)).y(d => y_signal(d)));

    g.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + graphHeight_signal + ")")
        .call(d3.axisBottom(x_signal));

    svg.append("g")
        .attr("class", "y axis")
        .call(d3.axisLeft(y_signal));

    function signal_zoom(transform) {
        currentZoomECG = transform

        d3.selectAll(".ecg").each(function () {
            const svg = d3.select(this);

            x_signal.range([0, graphWidth_signal]);
            const xz = transform.rescaleX(x_signal);
            
            svg.select(".x.axis").call(d3.axisBottom(xz));
            svg.select(".line").attr("d", d3.line()
                .x((d, i) => xz(i))
                .y(d => y_signal(d))
            );
        });
    }
}

// Function to get signal from data, and graph
function show_signal(d, data, principal=true) {
    d3.selectAll("#noneECG").style("display", "none");
    let ids = data.map(d => d.id)

    fetch('/get_signal', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json'},
        body: JSON.stringify({
            id: d.id,
            ids: ids,
            n: 2
        }),
    })
        .then(response => response.json())
        .then(result => {
            const { ids: signal_id, signals: signals } = result;

            if (principal){
                d3.selectAll("#principal .ecg").remove();
                draw_signal(d, signals[0], '#principal');
            }

            d3.selectAll(".dot").classed("nearby", false);
            if (signals.length > 1){
                d3.selectAll("#nearby .ecg").remove();
                for (let i = 1; i < signals.length; i++) {
                    id = signal_id[i]
                    d3.select("#dot-" + id).classed("nearby", true);
                    d = data.find(record => record.id === id);
                    draw_signal(d, signals[i], '#nearby');
                }
            }
        })
        .catch((error) => {
            console.log('Error:', error);
        });
}

function update(data) {
    let filtered_data = get_filter(data);

     // Check if any cluster is selected
    if (filtered_data.length === 0) {
        clearECGDisplays();
        svg.selectAll(".dot").remove();
        d3.select("#noneCluster").style("display", "block");
        d3.select("#scatter-plot").style("display", "none");
        return;
    }else{
        d3.select("#noneCluster").style("display", "none");
        d3.select("#scatter-plot").style("display", "block");
    }

    // Check if selected point is still in filtered data
    const selected_dot = d3.select(".dot.selected").data()[0];
    if (selected_dot) {
        const still = filtered_data.some(d => d.id === selected_dot.id);
        if (still) {
            show_signal(selected_dot, filtered_data, false)
        } else {
            clearECGDisplays();
        }
    }

    const dots = svg.selectAll(".dot").data(filtered_data, d => d.id);
    dots.exit().remove();
    dots.attr("cx", d => x(d.x))
        .attr("cy", d => y(d.y))
        .style("fill", d => cluster_color[d.cluster]);

    // Add new points
    dots.enter().append("circle")
        .attr("class", "dot")
        .attr("r", 3)
        .attr("cx", d => x(d.x))
        .attr("cy", d => y(d.y))
        .attr("id", d => "dot-" + d.id)
        .style("fill", d => cluster_color[d.cluster])
        .on("mouseover", function (event, d) {
            show_tooltip(event, d);
            d3.select("#ecg-" + d.id).select('.line').classed("highlight", true);  
        })
        .on("mouseout", function (event, d) {
            show_tooltip(null, null, false);
            d3.select("#ecg-" + d.id).select('.line').classed("highlight", false);
        })
        .on("click", function (event, d) {
            d3.selectAll(".dot.selected").classed("selected", false);
            d3.select(this).classed("selected", true);
            let filtered_data = get_filter(data);
            show_signal(d, filtered_data);
            // zoomToPoint(d);
        });
    applyZoomTransform(currentZoomState);
}


// Load Data from API
fetch('/get_data')
    .then(response => response.json())
    .then(data => {

        // Get global parameters: min age, max age, clusters.
        const min_age = d3.min(data, d => d.age);
        const max_age = d3.max(data, d => d.age);
        const clusters = [...new Set(data.map(d => d.cluster))];

        // Create slider for age range and buttons for each cluster
        const age_slider = create_sliders(min_age, max_age);
        create_buttons(clusters);

        // Define domain
        x.domain(d3.extent(data, d => d.x));
        y.domain(d3.extent(data, d => d.y));

        // Show and update
        update(data);

        // Check changes in filters
        d3.selectAll(".cluster-button").on("click", function () {
            let btn = d3.select(this);

            if (this.getAttribute("data-cluster") === "5") {
                if (btn.classed("selected")) {
                    // Deselect all clusters
                    d3.selectAll('.cluster-button').classed("selected", false);
                } else {
                    // Select all clusters
                    d3.selectAll('.cluster-button').classed("selected", true);
                }
            }else {
                // Toggle button selection
                btn.classed("selected", !btn.classed("selected"));
                size = d3.selectAll(".cluster-button:not([data-cluster='5']).selected").size();

                // If not all clusters are selected
                if (size < clusters.length) {
                    d3.select('.cluster-button[data-cluster="5"]').classed("selected", false);
                }
        
                // If all clusters are selected, so... ALL too.
                if (size === clusters.length) {
                    d3.select('.cluster-button[data-cluster="5"]').classed("selected", true);
                }
            }
            update(data);
        });

        d3.selectAll("#gender-icons i").on("click", function () {
            d3.selectAll("#gender-icons i").classed("selected", false);
            d3.select(this).classed("selected", true);
            update(data);
        });

        age_slider.noUiSlider.on('update', function (values, handle) {
            update(data);
            const selected_dot = d3.select(".dot.selected").data()[0];
            if(!selected_dot){
                clearECGDisplays();
            }
        });
    })
    .catch(error => {
        console.error('Error fetching data:', error);
    });
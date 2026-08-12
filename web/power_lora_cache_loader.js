// Adapted from rgthree-comfy's Power Lora Loader (MIT).
// Copyright (c) 2023 Regis Gaughan, III (rgthree).
// See THIRD_PARTY_NOTICES.md for the full MIT license notice.

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_TYPE = "FuseAndLoadQ8CRLoras";

function isLowQuality() {
  return (app.canvas.ds?.scale ?? 1) <= 0.5;
}

function fitString(ctx, text, maxWidth) {
  if (ctx.measureText(text).width <= maxWidth) return text;
  const ellipsis = "…";
  let low = 0;
  let high = text.length;
  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    if (ctx.measureText(text.substring(0, middle) + ellipsis).width <= maxWidth) {
      low = middle + 1;
    } else {
      high = middle - 1;
    }
  }
  return text.substring(0, high) + ellipsis;
}

function drawRoundedRectangle(ctx, { pos, size, borderRadius, colorStroke, colorBackground }) {
  const lowQuality = isLowQuality();
  ctx.save();
  ctx.strokeStyle = colorStroke ?? LiteGraph.WIDGET_OUTLINE_COLOR;
  ctx.fillStyle = colorBackground ?? LiteGraph.WIDGET_BGCOLOR;
  ctx.beginPath();
  ctx.roundRect(
    pos[0],
    pos[1],
    size[0],
    size[1],
    lowQuality ? [0] : [borderRadius ?? size[1] * 0.5],
  );
  ctx.fill();
  if (!lowQuality) ctx.stroke();
  ctx.restore();
}

function drawWidgetButton(ctx, options, text, pressed) {
  const borderRadius = isLowQuality() ? 0 : (options.borderRadius ?? 4);
  ctx.save();
  if (!isLowQuality() && !pressed) {
    drawRoundedRectangle(ctx, {
      size: [options.size[0] - 2, options.size[1]],
      pos: [options.pos[0] + 1, options.pos[1] + 1],
      borderRadius,
      colorBackground: "#000000aa",
      colorStroke: "#000000aa",
    });
  }
  drawRoundedRectangle(ctx, {
    size: options.size,
    pos: [options.pos[0], options.pos[1] + (pressed ? 1 : 0)],
    borderRadius,
    colorBackground: pressed ? "#444" : LiteGraph.WIDGET_BGCOLOR,
    colorStroke: "transparent",
  });
  if (!isLowQuality() && !pressed) {
    drawRoundedRectangle(ctx, {
      size: [options.size[0] - 0.75, options.size[1] - 0.75],
      pos: options.pos,
      borderRadius: borderRadius - 0.5,
      colorBackground: "transparent",
      colorStroke: "#00000044",
    });
    drawRoundedRectangle(ctx, {
      size: [options.size[0] - 0.75, options.size[1] - 0.75],
      pos: [options.pos[0] + 0.75, options.pos[1] + 0.75],
      borderRadius: borderRadius - 0.5,
      colorBackground: "transparent",
      colorStroke: "#ffffff11",
    });
  }
  drawRoundedRectangle(ctx, {
    size: options.size,
    pos: [options.pos[0], options.pos[1] + (pressed ? 1 : 0)],
    borderRadius,
    colorBackground: "transparent",
  });
  if (!isLowQuality() && text) {
    ctx.textBaseline = "middle";
    ctx.textAlign = "center";
    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
    ctx.fillText(text, options.size[0] / 2, options.pos[1] + options.size[1] / 2 + (pressed ? 1 : 0));
  }
  ctx.restore();
}

function drawNumberWidgetPart(ctx, { posX, posY, height, value }) {
  const arrowWidth = 9;
  const arrowHeight = 10;
  const innerMargin = 3;
  const numberWidth = 32;
  const leftArrow = [posX - arrowWidth - innerMargin - numberWidth - innerMargin - arrowWidth, arrowWidth];
  const text = [leftArrow[0] + arrowWidth + innerMargin, numberWidth];
  const rightArrow = [text[0] + numberWidth + innerMargin, arrowWidth];
  const midY = posY + height / 2;

  ctx.save();
  ctx.fill(new Path2D(`M ${leftArrow[0]} ${midY} l ${arrowWidth} ${arrowHeight / 2} l 0 -${arrowHeight} L ${leftArrow[0]} ${midY} z`));
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(fitString(ctx, Number(value).toFixed(2), numberWidth), text[0] + numberWidth / 2, midY);
  ctx.fill(new Path2D(`M ${rightArrow[0]} ${midY - arrowHeight / 2} l ${arrowWidth} ${arrowHeight / 2} l -${arrowWidth} ${arrowHeight / 2} v -${arrowHeight} z`));
  ctx.restore();
  return [leftArrow, text, rightArrow];
}

const NUMBER_WIDGET_WIDTH = 9 + 3 + 32 + 3 + 9;

function drawTogglePart(ctx, { posX, posY, height, value }) {
  const lowQuality = isLowQuality();
  const toggleRadius = height * 0.36;
  const toggleWidth = height * 1.5;
  ctx.save();
  if (!lowQuality) {
    ctx.beginPath();
    ctx.roundRect(posX + 4, posY + 4, toggleWidth - 8, height - 8, [height * 0.5]);
    ctx.globalAlpha = app.canvas.editor_alpha * 0.25;
    ctx.fillStyle = "rgba(255,255,255,0.45)";
    ctx.fill();
    ctx.globalAlpha = app.canvas.editor_alpha;
  }
  ctx.fillStyle = value === true ? "#89B" : "#888";
  const toggleX = lowQuality || value === false
    ? posX + height * 0.5
    : value === true
      ? posX + height
      : posX + height * 0.75;
  ctx.beginPath();
  ctx.arc(toggleX, posY + height * 0.5, toggleRadius, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
  return [posX, toggleWidth];
}

class PowerBaseWidget {
  constructor(name) {
    this.name = name;
    this.type = "custom";
    this.options = {};
    this.y = 0;
    this.last_y = 0;
    this.mouseDowned = null;
    this.isMouseDownedAndOver = false;
    this.hitAreas = {};
    this.downedHitAreasForMove = [];
    this.downedHitAreasForClick = [];
  }

  serializeValue() {
    return this.value;
  }

  clickWasWithinBounds(pos, bounds) {
    const xEnd = bounds[0] + (bounds.length > 2 ? bounds[2] : bounds[1]);
    if (pos[0] < bounds[0] || pos[0] > xEnd) return false;
    return bounds.length === 2 || (pos[1] >= bounds[1] && pos[1] <= bounds[1] + bounds[3]);
  }

  cancelMouseDown() {
    this.mouseDowned = null;
    this.isMouseDownedAndOver = false;
    this.downedHitAreasForMove.length = 0;
  }

  mouse(event, pos, node) {
    if (event.type === "pointerdown") {
      this.mouseDowned = [...pos];
      this.isMouseDownedAndOver = true;
      this.downedHitAreasForMove.length = 0;
      this.downedHitAreasForClick.length = 0;
      let handled = false;
      for (const area of Object.values(this.hitAreas)) {
        if (!this.clickWasWithinBounds(pos, area.bounds)) continue;
        if (area.onMove) this.downedHitAreasForMove.push(area);
        if (area.onClick) this.downedHitAreasForClick.push(area);
        if (area.onDown) handled = area.onDown.call(this, event, pos, node, area) === true || handled;
        area.wasMouseClickedAndIsOver = true;
      }
      return handled;
    }
    if (event.type === "pointerup") {
      if (!this.mouseDowned) return true;
      this.downedHitAreasForMove.length = 0;
      this.cancelMouseDown();
      let handled = false;
      for (const area of Object.values(this.hitAreas)) {
        if (area.onUp && this.clickWasWithinBounds(pos, area.bounds)) {
          handled = area.onUp.call(this, event, pos, node, area) === true || handled;
        }
        area.wasMouseClickedAndIsOver = false;
      }
      for (const area of this.downedHitAreasForClick) {
        if (this.clickWasWithinBounds(pos, area.bounds)) {
          handled = area.onClick.call(this, event, pos, node, area) === true || handled;
        }
      }
      this.downedHitAreasForClick.length = 0;
      return handled;
    }
    if (event.type === "pointermove") {
      for (const area of this.downedHitAreasForMove) area.onMove.call(this, event, pos, node, area);
      return !!this.mouseDowned;
    }
    return false;
  }
}

class DividerWidget extends PowerBaseWidget {
  constructor({ marginTop = 7, marginBottom = 7, thickness = 1 } = {}) {
    super("divider");
    this.value = {};
    this.options = { serialize: false };
    this.marginTop = marginTop;
    this.marginBottom = marginBottom;
    this.thickness = thickness;
  }

  draw(ctx, node, width, y) {
    if (!this.thickness) return;
    ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
    ctx.stroke(new Path2D(`M 15 ${y + this.marginTop} h ${node.size[0] - 30}`));
  }

  computeSize(width) {
    return [width, this.marginTop + this.marginBottom + this.thickness];
  }
}

class AddLoraButtonWidget extends PowerBaseWidget {
  constructor(onClick) {
    super("➕ Add Lora");
    this.value = "";
    this.options = { serialize: false };
    this.onClick = onClick;
  }

  draw(ctx, node, width, y, height) {
    drawWidgetButton(ctx, { size: [node.size[0] - 30, height], pos: [15, y] }, this.name, this.isMouseDownedAndOver);
  }

  mouse(event, pos, node) {
    if (event.type === "pointerdown") {
      this.mouseDowned = [...pos];
      this.isMouseDownedAndOver = true;
      return true;
    }
    if (event.type === "pointerup" && this.mouseDowned) {
      this.cancelMouseDown();
      return this.onClick(event, pos, node);
    }
    return event.type === "pointermove";
  }
}

class HeaderWidget extends PowerBaseWidget {
  constructor() {
    super("power_lora_header");
    this.value = {};
    this.options = { serialize: false };
    this.hitAreas.toggle = { bounds: [0, 0], onDown: this.onToggleDown };
  }

  draw(ctx, node, width, y, height) {
    if (!node.hasPowerLoraRows()) return;
    const margin = 10;
    const innerMargin = margin * 0.33;
    const midY = y + 2 + height * 0.5;
    let posX = margin;
    ctx.save();
    this.hitAreas.toggle.bounds = drawTogglePart(ctx, { posX, posY: y + 2, height, value: node.allPowerLorasState() });
    if (!isLowQuality()) {
      posX += this.hitAreas.toggle.bounds[1] + innerMargin;
      ctx.globalAlpha = app.canvas.editor_alpha * 0.55;
      ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText("Toggle All", posX, midY);
      const rightX = node.size[0] - margin - innerMargin * 2;
      ctx.textAlign = "center";
      ctx.fillText("Strength", rightX - NUMBER_WIDGET_WIDTH / 2, midY);
    }
    ctx.restore();
  }

  onToggleDown(event, pos, node) {
    node.toggleAllPowerLoras();
    this.cancelMouseDown();
    node.setDirtyCanvas(true, true);
    return true;
  }
}

class LoraRowWidget extends PowerBaseWidget {
  constructor(name, loras, value) {
    super(name);
    this.loras = loras;
    this.value = {
      on: value?.on ?? true,
      lora: value?.lora ?? null,
      strength: value?.strength ?? 1,
    };
    this.hitAreas = {
      toggle: { bounds: [0, 0], onDown: this.onToggleDown },
      lora: { bounds: [0, 0], onClick: this.onLoraClick },
      strengthDec: { bounds: [0, 0], onClick: this.onStrengthDec },
      strengthVal: { bounds: [0, 0], onClick: this.onStrengthValue },
      strengthInc: { bounds: [0, 0], onClick: this.onStrengthInc },
      strengthAny: { bounds: [0, 0], onMove: this.onStrengthMove },
    };
    this.haveMouseMovedStrength = false;
  }

  draw(ctx, node, width, y, height) {
    const margin = 10;
    const innerMargin = margin * 0.33;
    const midY = y + height * 0.5;
    let posX = margin;
    ctx.save();
    drawRoundedRectangle(ctx, { pos: [posX, y], size: [node.size[0] - margin * 2, height] });
    this.hitAreas.toggle.bounds = drawTogglePart(ctx, { posX, posY: y, height, value: this.value.on });
    posX += this.hitAreas.toggle.bounds[1] + innerMargin;
    if (isLowQuality()) {
      ctx.restore();
      return;
    }
    if (!this.value.on) ctx.globalAlpha = app.canvas.editor_alpha * 0.4;
    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
    const rightX = node.size[0] - margin - innerMargin * 2;
    const [leftArrow, text, rightArrow] = drawNumberWidgetPart(ctx, {
      posX: rightX,
      posY: y,
      height,
      value: this.value.strength ?? 1,
    });
    this.hitAreas.strengthDec.bounds = leftArrow;
    this.hitAreas.strengthVal.bounds = text;
    this.hitAreas.strengthInc.bounds = rightArrow;
    this.hitAreas.strengthAny.bounds = [leftArrow[0], rightArrow[0] + rightArrow[1] - leftArrow[0]];
    const loraWidth = leftArrow[0] - innerMargin - posX;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(fitString(ctx, String(this.value.lora || "None"), loraWidth), posX, midY);
    this.hitAreas.lora.bounds = [posX, loraWidth];
    ctx.globalAlpha = app.canvas.editor_alpha;
    ctx.restore();
  }

  serializeValue() {
    return { ...this.value };
  }

  onToggleDown() {
    this.value.on = !this.value.on;
    this.cancelMouseDown();
    return true;
  }

  onLoraClick(event, pos, node) {
    new LiteGraph.ContextMenu(this.loras, {
      event,
      title: "Choose a lora",
      scale: Math.max(1, app.canvas.ds?.scale ?? 1),
      className: "dark",
      callback: (value) => {
        if (typeof value === "string") this.value.lora = value;
        node.setDirtyCanvas(true, true);
      },
    });
    this.cancelMouseDown();
    return true;
  }

  onStrengthDec() {
    this.stepStrength(-1);
    return true;
  }

  onStrengthInc() {
    this.stepStrength(1);
    return true;
  }

  onStrengthMove(event) {
    if (!event.deltaX) return;
    this.haveMouseMovedStrength = true;
    this.value.strength = (this.value.strength ?? 1) + event.deltaX * 0.05;
  }

  onStrengthValue(event, pos, node) {
    if (!this.haveMouseMovedStrength) {
      app.canvas.prompt("Value", this.value.strength, (value) => {
        this.value.strength = Number(value);
        node.setDirtyCanvas(true, true);
      }, event);
    }
    this.haveMouseMovedStrength = false;
    return true;
  }

  stepStrength(direction) {
    this.value.strength = Math.round(((this.value.strength ?? 1) + direction * 0.05) * 100) / 100;
  }
}

app.registerExtension({
  name: "ComfyUI-GGUF.PowerLoRACacheLoader",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;
    const loraInfo = await api.fetchApi("/object_info/LoraLoader").then((response) => response.json());
    const loras = loraInfo.LoraLoader.input.required.lora_name[0];

    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      originalCreated?.apply(this, arguments);
      this._powerRows = [];
      this._powerLoraCounter = 0;
      this.addCustomWidget(new DividerWidget({ marginTop: 4, marginBottom: 0, thickness: 0 }));
      this.addCustomWidget(new HeaderWidget());
      this._powerButtonSpacer = this.addCustomWidget(new DividerWidget({ marginTop: 4, marginBottom: 0, thickness: 0 }));
      this.addCustomWidget(new AddLoraButtonWidget((event) => {
        new LiteGraph.ContextMenu(loras, {
          event,
          title: "Choose a lora",
          scale: Math.max(1, app.canvas.ds?.scale ?? 1),
          className: "dark",
          callback: (value) => {
            if (typeof value === "string") {
              this.addPowerLoraRow(value);
              this.setSize(this.computeSize());
              this.setDirtyCanvas(true, true);
            }
          },
        });
        return true;
      }));
    };

    nodeType.prototype.hasPowerLoraRows = function () {
      return this._powerRows.length > 0;
    };

    nodeType.prototype.allPowerLorasState = function () {
      const states = this._powerRows.map((row) => row.value.on);
      return states.every(Boolean) ? true : states.every((state) => !state) ? false : null;
    };

    nodeType.prototype.toggleAllPowerLoras = function () {
      const nextValue = this.allPowerLorasState() !== true;
      for (const row of this._powerRows) row.value.on = nextValue;
    };

    nodeType.prototype.addPowerLoraRow = function (lora = null, value = null) {
      const row = this.addCustomWidget(new LoraRowWidget(
        `lora_${++this._powerLoraCounter}`,
        loras,
        value ?? { lora },
      ));
      const rowIndex = this.widgets.indexOf(row);
      const spacerIndex = this.widgets.indexOf(this._powerButtonSpacer);
      this.widgets.splice(rowIndex, 1);
      this.widgets.splice(spacerIndex, 0, row);
      this._powerRows.push(row);
      return row;
    };

    const originalConfigure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      originalConfigure?.apply(this, arguments);
      for (const row of this._powerRows) {
        const index = this.widgets.indexOf(row);
        if (index >= 0) this.widgets.splice(index, 1);
      }
      this._powerRows = [];
      this._powerLoraCounter = 0;
      const savedRows = (info.widgets_values ?? []).filter(
        (value) => value && typeof value === "object" && "lora" in value,
      );
      for (const value of savedRows) this.addPowerLoraRow(null, value);
      this.size[1] = Math.max(this.size[1], this.computeSize()[1]);
    };
  },
});

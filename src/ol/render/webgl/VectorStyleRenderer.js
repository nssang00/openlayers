/**
 * @module ol/render/webgl/VectorStyleRenderer
 */
import {
  create as createTransform,
  makeInverse as makeInverseTransform,
} from '../../transform.js';
import WebGLArrayBuffer from '../../webgl/Buffer.js';
import {AttributeType} from '../../webgl/Helper.js';
import {ARRAY_BUFFER, DYNAMIC_DRAW, ELEMENT_ARRAY_BUFFER} from '../../webgl.js';
import {create as createWebGLWorker} from '../../worker/webgl.js';
import {WebGLWorkerMessageType} from './constants.js';
import {colorEncodeId} from './encodeUtil.js';
import {
  generateLineStringRenderInstructions,
  generatePointRenderInstructions,
  generatePolygonRenderInstructions,
  getCustomAttributesSize,
} from './renderinstructions.js';
import {parseLiteralStyle} from './style.js';

//kmg
import {buildExpression, newEvaluationContext} from '../../expr/cpu.js';
import {
  BooleanType,
  computeGeometryType,
  newParsingContext,
} from '../../expr/expression.js';
import earcut from 'earcut';
import {inflateEnds} from '../../geom/flat/orient.js';
import {UNDEFINED_PROP_VALUE} from '../../expr/gpu.js';
import {transform2D} from '../../geom/flat/transform.js';
import {apply as applyTransform} from '../../transform.js';

const tmpColor = [];
/** @type {Worker|undefined} */
let WEBGL_WORKER;
function getWebGLWorker() {
  if (!WEBGL_WORKER) {
    WEBGL_WORKER = createWebGLWorker();
  }
  return WEBGL_WORKER;
}
let workerMessageCounter = 0;

/**
 * Names of attributes made available to the vertex shader.
 * Please note: changing these *will* break custom shaders!
 * @enum {string}
 */
export const Attributes = {
  POSITION: 'a_position',
  LOCAL_POSITION: 'a_localPosition',
  SEGMENT_START: 'a_segmentStart',
  SEGMENT_END: 'a_segmentEnd',
  MEASURE_START: 'a_measureStart',
  MEASURE_END: 'a_measureEnd',
  ANGLE_TANGENT_SUM: 'a_angleTangentSum',
  JOIN_ANGLES: 'a_joinAngles',
  DISTANCE: 'a_distance',
};

/**
 * @typedef {Object} AttributeDefinition A description of a custom attribute to be passed on to the GPU, with a value different
 * for each feature.
 * @property {number} [size] Amount of numerical values composing the attribute, either 1, 2, 3 or 4; in case size is > 1, the return value
 * of the callback should be an array; if unspecified, assumed to be a single float value
 * @property {function(this:import("./MixedGeometryBatch.js").GeometryBatchItem, import("../../Feature").FeatureLike):number|Array<number>} callback This callback computes the numerical value of the
 * attribute for a given feature.
 */

/**
 * @typedef {Object<string, AttributeDefinition>} AttributeDefinitions
 * @typedef {Object<string, import("../../webgl/Helper").UniformValue>} UniformDefinitions
 */

/**
 * @typedef {Array<WebGLArrayBuffer>} WebGLArrayBufferSet Buffers organized like so: [indicesBuffer, vertexAttributesBuffer, instanceAttributesBuffer]
 */

/**
 * @typedef {Object} WebGLBuffers
 * @property {WebGLArrayBufferSet} polygonBuffers Array containing indices and vertices buffers for polygons
 * @property {WebGLArrayBufferSet} lineStringBuffers Array containing indices and vertices buffers for line strings
 * @property {WebGLArrayBufferSet} pointBuffers Array containing indices and vertices buffers for points
 * @property {import("../../transform.js").Transform} invertVerticesTransform Inverse of the transform applied when generating buffers
 */

/**
 * @typedef {Object} RenderInstructions
 * @property {Float32Array|null} polygonInstructions Polygon instructions; null if nothing to render
 * @property {Float32Array|null} lineStringInstructions LineString instructions; null if nothing to render
 * @property {Float32Array|null} pointInstructions Point instructions; null if nothing to render
 */

/**
 * @typedef {Object} ShaderProgram An object containing both shaders (vertex and fragment)
 * @property {string} vertex Vertex shader source
 * @property {string} fragment Fragment shader source
 */

/**
 * @typedef {import('./style.js').StyleParseResult} StyleShaders
 */

/**
 * @typedef {import('../../style/flat.js').FlatStyleLike} FlatStyleLike
 */
/**
 * @typedef {import('../../style/flat.js').FlatStyle} FlatStyle
 */
/**
 * @typedef {import('../../style/flat.js').Rule} FlatStyleRule
 */

/**
 * @typedef {Object} SubRenderPass
 * @property {string} vertexShader Vertex shader
 * @property {string} fragmentShader Fragment shader
 * @property {Array<import('../../webgl/Helper.js').AttributeDescription>} attributesDesc Attributes description, defined for each primitive vertex
 * @property {Array<import('../../webgl/Helper.js').AttributeDescription>} instancedAttributesDesc Attributes description, defined once per primitive
 * @property {number} instancePrimitiveVertexCount Number of vertices per instance primitive in this render pass
 * @property {WebGLProgram} [program] Program; this has to be recreated if the helper is lost/changed
 */

/**
 * @typedef {Object} RenderPass
 * @property {SubRenderPass} [fillRenderPass] Fill render pass; undefined if no fill in pass
 * @property {SubRenderPass} [strokeRenderPass] Stroke render pass; undefined if no stroke in pass
 * @property {SubRenderPass} [symbolRenderPass] Symbol render pass; undefined if no symbol in pass
 */

/**
 * @classdesc This class is responsible for:
 * 1. generating WebGL buffers according to a provided style, using a MixedGeometryBatch as input
 * 2. rendering geometries contained in said buffers
 *
 * A VectorStyleRenderer instance can be created either from a literal style or from shaders.
 * The shaders should not be provided explicitly but instead as a preconfigured ShaderBuilder instance.
 *
 * The `generateBuffers` method returns a promise resolving to WebGL buffers that are intended to be rendered by the
 * same renderer.
 */
class VectorStyleRenderer {
  /**
   * @param {FlatStyleLike|StyleShaders|Array<StyleShaders>} styles Vector styles expressed as flat styles, flat style rules or style shaders
   * @param {import('../../style/flat.js').StyleVariables} variables Style variables
   * @param {import('../../webgl/Helper.js').default} helper Helper
   * @param {boolean} [enableHitDetection] Whether to enable the hit detection (needs compatible shader)
   */
  constructor(styles, variables, helper, enableHitDetection, enableWorker) {//kmg
    /**
     * @private
     * @type {import('../../webgl/Helper.js').default}
     */
    this.helper_;

    /**
     * @private
     */
    this.hitDetectionEnabled_ = !!enableHitDetection;

    /**
     * @private
     */
    this.workerEnabled_ = !!enableWorker;
    
    /**
     * @type {Array<StyleShaders>}
     * @private
     */
    this.styleShaders = convertStyleToShaders(styles, variables);

    /**
     * @type {AttributeDefinitions}
     * @private
     */
    this.customAttributes_ = {};

    /**
     @type {UniformDefinitions}
     * @private
     */
    this.uniforms_ = {};

    // add hit detection attribute if enabled
    if (this.hitDetectionEnabled_) {
      this.customAttributes_['hitColor'] = {
        callback() {
          return colorEncodeId(this.ref, tmpColor);
        },
        size: 4,
      };
    }

    // add attributes & uniforms coming from all shaders
    for (const styleShader of this.styleShaders) {
      for (const attributeName in styleShader.attributes) {
        if (attributeName in this.customAttributes_) {
          // already defined: skip
          continue;
        }
        this.customAttributes_[attributeName] =
          styleShader.attributes[attributeName];
      }
      for (const uniformName in styleShader.uniforms) {
        if (uniformName in this.uniforms_) {
          // already defined: skip
          continue;
        }
        this.uniforms_[uniformName] = styleShader.uniforms[uniformName];
      }
    }

    // create a render pass for each shader
    /**
     * @type {Array<RenderPass>}
     * @private
     */
    this.renderPasses_ = this.styleShaders.map((styleShader) => {
      /** @type {RenderPass} */
      const renderPass = {};

      const customAttributesDesc = Object.entries(this.customAttributes_).map(
        ([name, value]) => {
          const isUsed = name in styleShader.attributes || name === 'hitColor';
          return {
            name: isUsed ? `a_${name}` : null, // giving a null name means this is only used for "spacing" in between attributes
            size: value.size || 1,
            type: AttributeType.FLOAT,
          };
        },
      );

      // set up each subpass
      if (styleShader.builder.getFillVertexShader()) {
        renderPass.fillRenderPass = {
          vertexShader: styleShader.builder.getFillVertexShader(),
          fragmentShader: styleShader.builder.getFillFragmentShader(),
          attributesDesc: [
            {
              name: Attributes.POSITION,
              size: 2,
              type: AttributeType.FLOAT,
            },
            ...customAttributesDesc,
          ],
          instancedAttributesDesc: [], // no instanced rendering for polygons
          instancePrimitiveVertexCount: 3,
        };
      }
      if (styleShader.builder.getStrokeVertexShader()) {
        renderPass.strokeRenderPass = {
          vertexShader: styleShader.builder.getStrokeVertexShader(),
          fragmentShader: styleShader.builder.getStrokeFragmentShader(),
          attributesDesc: [
            {
              name: Attributes.LOCAL_POSITION,
              size: 2,
              type: AttributeType.FLOAT,
            },
          ],
          instancedAttributesDesc: [
            {
              name: Attributes.SEGMENT_START,
              size: 2,
              type: AttributeType.FLOAT,
            },
            {
              name: Attributes.MEASURE_START,
              size: 1,
              type: AttributeType.FLOAT,
            },
            {
              name: Attributes.SEGMENT_END,
              size: 2,
              type: AttributeType.FLOAT,
            },
            {
              name: Attributes.MEASURE_END,
              size: 1,
              type: AttributeType.FLOAT,
            },
            {
              name: Attributes.JOIN_ANGLES,
              size: 2,
              type: AttributeType.FLOAT,
            },
            {
              name: Attributes.DISTANCE,
              size: 1,
              type: AttributeType.FLOAT,
            },
            {
              name: Attributes.ANGLE_TANGENT_SUM,
              size: 1,
              type: AttributeType.FLOAT,
            },
            ...customAttributesDesc,
          ],
          instancePrimitiveVertexCount: 6,
        };
      }
      if (styleShader.builder.getSymbolVertexShader()) {
        renderPass.symbolRenderPass = {
          vertexShader: styleShader.builder.getSymbolVertexShader(),
          fragmentShader: styleShader.builder.getSymbolFragmentShader(),
          attributesDesc: [
            {
              name: Attributes.LOCAL_POSITION,
              size: 2,
              type: AttributeType.FLOAT,
            },
          ],
          instancedAttributesDesc: [
            {
              name: Attributes.POSITION,
              size: 2,
              type: AttributeType.FLOAT,
            },
            ...customAttributesDesc,
          ],
          instancePrimitiveVertexCount: 6,
        };
      }
      //kmg
      if(styleShader.contextFilter)
        renderPass.contextFilter = styleShader.contextFilter;
      return renderPass;
    });

    this.hasFill_ = this.renderPasses_.some((pass) => pass.fillRenderPass);
    this.hasStroke_ = this.renderPasses_.some((pass) => pass.strokeRenderPass);
    this.hasSymbol_ = this.renderPasses_.some((pass) => pass.symbolRenderPass);

    // this will initialize render passes with the given helper
    this.setHelper(helper);
  }

  /**
   * @param {import('./MixedGeometryBatch.js').default} geometryBatch Geometry batch
   * @param {import("../../transform.js").Transform} transform Transform to apply to coordinates
   * @return {Promise<WebGLBuffers|null>} A promise resolving to WebGL buffers; returns null if buffers are empty
   */
  async generateBuffers(geometryBatch, transform) {
    if (geometryBatch.isEmpty()) {
      return null;
    }
    const renderInstructions = this.generateRenderInstructions_(
      geometryBatch,
      transform,
    );
    const [polygonBuffers, lineStringBuffers, pointBuffers] = await Promise.all(
      [
        this.generateBuffersForType_(
          renderInstructions.polygonInstructions,
          'Polygon',
          transform,
        ),
        this.generateBuffersForType_(
          renderInstructions.lineStringInstructions,
          'LineString',
          transform,
        ),
        this.generateBuffersForType_(
          renderInstructions.pointInstructions,
          'Point',
          transform,
        ),
      ],
    );
    // also return the inverse of the transform that was applied when generating buffers
    const invertVerticesTransform = makeInverseTransform(
      createTransform(),
      transform,
    );
    return {
      polygonBuffers: polygonBuffers,
      lineStringBuffers: lineStringBuffers,
      pointBuffers: pointBuffers,
      invertVerticesTransform: invertVerticesTransform,
    };
  }

  /**
   * @param {import('./MixedGeometryBatch.js').default} geometryBatch Geometry batch
   * @param {import("../../transform.js").Transform} transform Transform to apply to coordinates
   * @return {RenderInstructions} Render instructions
   * @private
   */
  generateRenderInstructions_(geometryBatch, transform) {
    const polygonInstructions = this.hasFill_
      ? generatePolygonRenderInstructions(
          geometryBatch.polygonBatch,
          new Float32Array(0),
          this.customAttributes_,
          transform,
        )
      : null;
    const lineStringInstructions = this.hasStroke_
      ? generateLineStringRenderInstructions(
          geometryBatch.lineStringBatch,
          new Float32Array(0),
          this.customAttributes_,
          transform,
        )
      : null;
    const pointInstructions = this.hasSymbol_
      ? generatePointRenderInstructions(
          geometryBatch.pointBatch,
          new Float32Array(0),
          this.customAttributes_,
          transform,
        )
      : null;

    return {
      polygonInstructions,
      lineStringInstructions,
      pointInstructions,
    };
  }

  /**
   * @param {Float32Array|null} renderInstructions Render instructions
   * @param {import("../../geom/Geometry.js").Type} geometryType Geometry type
   * @param {import("../../transform.js").Transform} transform Transform to apply to coordinates
   * @return {Promise<WebGLArrayBufferSet>|null} Indices buffer and vertices buffer; null if nothing to render
   * @private
   */
  generateBuffersForType_(renderInstructions, geometryType, transform) {
    if (renderInstructions === null) {
      return null;
    }

    const messageId = workerMessageCounter++;
    let messageType;
    switch (geometryType) {
      case 'Polygon':
        messageType = WebGLWorkerMessageType.GENERATE_POLYGON_BUFFERS;
        break;
      case 'LineString':
        messageType = WebGLWorkerMessageType.GENERATE_LINE_STRING_BUFFERS;
        break;
      case 'Point':
        messageType = WebGLWorkerMessageType.GENERATE_POINT_BUFFERS;
        break;
      default:
      // pass
    }

    /** @type {import('./constants.js').WebGLWorkerGenerateBuffersMessage} */
    const message = {
      id: messageId,
      type: messageType,
      renderInstructions: renderInstructions.buffer,
      renderInstructionsTransform: transform,
      customAttributesSize: getCustomAttributesSize(this.customAttributes_),
    };
    const WEBGL_WORKER = getWebGLWorker();
    WEBGL_WORKER.postMessage(message, [renderInstructions.buffer]);

    // leave ownership of render instructions
    renderInstructions = null;

    return new Promise((resolve) => {
      /**
       * @param {{data: import('./constants.js').WebGLWorkerGenerateBuffersMessage}} event Event.
       */
      const handleMessage = (event) => {
        const received = event.data;

        // this is not the response to our request: skip
        if (received.id !== messageId) {
          return;
        }

        // we've received our response: stop listening
        WEBGL_WORKER.removeEventListener('message', handleMessage);

        // the helper has disposed in the meantime; the promise will not be resolved
        if (!this.helper_.getGL()) {
          return;
        }

        // copy & flush received buffers to GPU
        const indicesBuffer = new WebGLArrayBuffer(
          ELEMENT_ARRAY_BUFFER,
          DYNAMIC_DRAW,
        ).fromArrayBuffer(received.indicesBuffer);
        const vertexAttributesBuffer = new WebGLArrayBuffer(
          ARRAY_BUFFER,
          DYNAMIC_DRAW,
        ).fromArrayBuffer(received.vertexAttributesBuffer);
        const instanceAttributesBuffer = new WebGLArrayBuffer(
          ARRAY_BUFFER,
          DYNAMIC_DRAW,
        ).fromArrayBuffer(received.instanceAttributesBuffer);
        this.helper_.flushBufferData(indicesBuffer);
        this.helper_.flushBufferData(vertexAttributesBuffer);
        this.helper_.flushBufferData(instanceAttributesBuffer);

        resolve([
          indicesBuffer,
          vertexAttributesBuffer,
          instanceAttributesBuffer,
        ]);
      };

      WEBGL_WORKER.addEventListener('message', handleMessage);
    });
  }

  /**
   * Render the geometries in the given buffers.
   * @param {WebGLBuffers} buffers WebGL Buffers to draw
   * @param {import("../../Map.js").FrameState} frameState Frame state
   * @param {function(): void} preRenderCallback This callback will be called right before drawing, and can be used to set uniforms
   */
  render(buffers, frameState, preRenderCallback) {
    for (const renderPass of this.renderPasses_) {
      //kmg
      if (renderPass.contextFilter) {
        if(!renderPass.contextFilter(frameState.viewState.resolution)) {
          continue;
        }
      }

      renderPass.fillRenderPass &&
        this.renderInternal_(
          buffers.polygonBuffers[0],
          buffers.polygonBuffers[1],
          buffers.polygonBuffers[2],
          renderPass.fillRenderPass,
          frameState,
          preRenderCallback,
        );
      renderPass.strokeRenderPass &&
        this.renderInternal_(
          buffers.lineStringBuffers[0],
          buffers.lineStringBuffers[1],
          buffers.lineStringBuffers[2],
          renderPass.strokeRenderPass,
          frameState,
          preRenderCallback,
        );
      renderPass.symbolRenderPass &&
        this.renderInternal_(
          buffers.pointBuffers[0],
          buffers.pointBuffers[1],
          buffers.pointBuffers[2],
          renderPass.symbolRenderPass,
          frameState,
          preRenderCallback,
        );
    }
  }

  /**
   * @param {WebGLArrayBuffer} indicesBuffer Indices buffer
   * @param {WebGLArrayBuffer} vertexAttributesBuffer Vertex attributes buffer
   * @param {WebGLArrayBuffer} instanceAttributesBuffer Instance attributes buffer
   * @param {SubRenderPass} subRenderPass Render pass (program, attributes, etc.) specific to one geometry type
   * @param {import("../../Map.js").FrameState} frameState Frame state.
   * @param {function(): void} preRenderCallback This callback will be called right before drawing, and can be used to set uniforms
   * @private
   */
  renderInternal_(
    indicesBuffer,
    vertexAttributesBuffer,
    instanceAttributesBuffer,
    subRenderPass,
    frameState,
    preRenderCallback,
  ) {
    const renderCount = indicesBuffer.getSize();
    if (renderCount === 0) {
      return;
    }

    const usesInstancedRendering = subRenderPass.instancedAttributesDesc.length;

    this.helper_.useProgram(subRenderPass.program, frameState);
    this.helper_.bindBuffer(vertexAttributesBuffer);
    this.helper_.bindBuffer(indicesBuffer);
    this.helper_.enableAttributes(subRenderPass.attributesDesc);
    this.helper_.bindBuffer(instanceAttributesBuffer);
    this.helper_.enableAttributesInstanced(
      subRenderPass.instancedAttributesDesc,
    );

    preRenderCallback();

    if (usesInstancedRendering) {
      const instanceAttributesStride =
        subRenderPass.instancedAttributesDesc.reduce(
          (prev, curr) => prev + (curr.size || 1),
          0,
        );
      const instanceCount =
        instanceAttributesBuffer.getSize() / instanceAttributesStride;

      this.helper_.drawElementsInstanced(0, renderCount, instanceCount);
    } else {
      this.helper_.drawElements(0, renderCount);
    }
  }

  /**
   * @param {import('../../webgl/Helper.js').default} helper Helper
   * @param {WebGLBuffers} buffers WebGL Buffers to reload if any
   */
  setHelper(helper, buffers = null) {
    this.helper_ = helper;

    for (const renderPass of this.renderPasses_) {
      if (renderPass.fillRenderPass) {
        renderPass.fillRenderPass.program = this.helper_.getProgram(
          renderPass.fillRenderPass.fragmentShader,
          renderPass.fillRenderPass.vertexShader,
        );
      }
      if (renderPass.strokeRenderPass) {
        renderPass.strokeRenderPass.program = this.helper_.getProgram(
          renderPass.strokeRenderPass.fragmentShader,
          renderPass.strokeRenderPass.vertexShader,
        );
      }
      if (renderPass.symbolRenderPass) {
        renderPass.symbolRenderPass.program = this.helper_.getProgram(
          renderPass.symbolRenderPass.fragmentShader,
          renderPass.symbolRenderPass.vertexShader,
        );
      }
    }
    this.helper_.addUniforms(this.uniforms_);

    if (buffers) {
      if (buffers.polygonBuffers) {
        this.helper_.flushBufferData(buffers.polygonBuffers[0]);
        this.helper_.flushBufferData(buffers.polygonBuffers[1]);
        this.helper_.flushBufferData(buffers.polygonBuffers[2]);
      }
      if (buffers.lineStringBuffers) {
        this.helper_.flushBufferData(buffers.lineStringBuffers[0]);
        this.helper_.flushBufferData(buffers.lineStringBuffers[1]);
        this.helper_.flushBufferData(buffers.lineStringBuffers[2]);
      }
      if (buffers.pointBuffers) {
        this.helper_.flushBufferData(buffers.pointBuffers[0]);
        this.helper_.flushBufferData(buffers.pointBuffers[1]);
        this.helper_.flushBufferData(buffers.pointBuffers[2]);
      }
    }
  }

  //kmg
  async generateBuffersFromFeatures(features, transform) {
    
    const filteredFeatures = [];

    const featureIdSet = new Set();
    for (const styleShader of this.styleShaders) {
      const filtered = styleShader.featureFilter
        ? features.filter(styleShader.featureFilter)
        : features;

      for (const feature of filtered) {
        const featureId = feature.getId() || feature.ol_uid;
        if (!featureIdSet.has(featureId)) {
          featureIdSet.add(featureId);
          filteredFeatures.push(feature);
        }
      }
    }

    const featuresBatch = {
      polygonFeatures: [],
      lineStringFeatures: [],
      pointFeatures: []
    };

    for(const feature of filteredFeatures) {
      const geometryType = feature.getGeometry().getType();
      if(geometryType === 'Polygon' || geometryType === 'MultiPolygon') {
        featuresBatch.polygonFeatures.push(feature);
        if(this.hasStroke_)
          featuresBatch.lineStringFeatures.push(feature);
      } else if(geometryType === 'LineString' || geometryType === 'MultiLineString') {
        featuresBatch.lineStringFeatures.push(feature);
      } else if(geometryType === 'Point' || geometryType === 'MultiPoint') {
        featuresBatch.pointFeatures.push(feature);
      } else {

      }   
    }
    if (featuresBatch.polygonFeatures.length === 0 && 
      featuresBatch.lineStringFeatures.length === 0 && 
      featuresBatch.pointFeatures.length === 0) {
      return null;
    }

    const renderInstructions = this.generateRenderInstructionsFromFeatures_(featuresBatch, transform);

    const label = `generateBuffersForType2_-${Date.now()}`;
    console.time(label);

    const polygonBuffers = this.generateWebGLBuffersFromInstructions_(
          renderInstructions.polygonInstructions,
          'Polygon', transform);

    const lineStringBuffers = this.generateWebGLBuffersFromInstructions_(
          renderInstructions.lineStringInstructions,
          'LineString', transform);

    const pointBuffers = this.generateWebGLBuffersFromInstructions_(
          renderInstructions.pointInstructions,
          'Point', transform);


    const invertVerticesTransform = makeInverseTransform(createTransform(), transform);
    console.timeEnd(label);
    
    return {
      polygonBuffers: polygonBuffers,
      lineStringBuffers: lineStringBuffers,
      pointBuffers: pointBuffers,
      invertVerticesTransform: invertVerticesTransform,
    };
  }

  generateWebGLBuffersFromInstructions_(renderInstructions, geometryType, transform) {
    if (!renderInstructions) return null;

    const customAttributesSize = getCustomAttributesSize(this.customAttributes_);
    let buffers;
    switch (geometryType) {
      case 'Point':
        buffers = generatePointBuffers_(renderInstructions, customAttributesSize);
        break;
      case 'LineString':
        buffers = generateLineStringBuffers_(renderInstructions, customAttributesSize, transform);
        break;
      case 'Polygon':
        buffers = generatePolygonBuffers_(renderInstructions, customAttributesSize);
        break;
      default:
        break;
    }

    const indicesBuffer = new WebGLArrayBuffer(ELEMENT_ARRAY_BUFFER, DYNAMIC_DRAW);
    indicesBuffer.setArray(buffers.indicesBuffer);
    const vertexAttributesBuffer = new WebGLArrayBuffer(ARRAY_BUFFER, DYNAMIC_DRAW);
    vertexAttributesBuffer.setArray(buffers.vertexAttributesBuffer);
    const instanceAttributesBuffer = new WebGLArrayBuffer(ARRAY_BUFFER, DYNAMIC_DRAW)
    instanceAttributesBuffer.setArray(buffers.instanceAttributesBuffer);

    this.helper_.flushBufferData(indicesBuffer);
    this.helper_.flushBufferData(vertexAttributesBuffer);
    this.helper_.flushBufferData(instanceAttributesBuffer);

    return [
      indicesBuffer,
      vertexAttributesBuffer,
      instanceAttributesBuffer,
    ];
  }

  generateRenderInstructionsFromFeatures_(featuresBatch, transform) {
    const polygonInstructions = this.hasFill_
      ? generatePolygonRenderInstructionsFromFeatures(
          featuresBatch.polygonFeatures,
          new Float32Array(0),
          this.customAttributes_,
          transform,
        )
      : null;
    const lineStringInstructions = this.hasStroke_
      ? generateLineStringRenderInstructionsFromFeatures(
          featuresBatch.lineStringFeatures,
          new Float32Array(0),
          this.customAttributes_,
          transform,
        )
      : null;
    const pointInstructions = this.hasSymbol_
      ? generatePointRenderInstructionsFromFeatures(
          featuresBatch.pointFeatures,
          new Float32Array(0),
          this.customAttributes_,
          transform,
        )
      : null;

    return {
      polygonInstructions,
      lineStringInstructions,
      pointInstructions,
    };
  }      

}

export default VectorStyleRenderer;

/**
 * Breaks down a vector style into an array of prebuilt shader builders with attributes and uniforms
 * @param {FlatStyleLike|StyleShaders|Array<StyleShaders>} style Vector style
 * @param {import('../../style/flat.js').StyleVariables} variables Style variables
 * @return {Array<StyleShaders>} Array of style shaders
 */
export function convertStyleToShaders(style, variables) {
  // possible cases:
  // - single shader
  // - multiple shaders
  // - single style
  // - multiple styles
  // - multiple rules
  const asArray = Array.isArray(style) ? style : [style];

  // if array of rules: break rules into separate styles, compute "else" filters
  if ('style' in asArray[0]) {
    /** @type {Array<StyleShaders>} */
    const shaders = [];
    const rules = /** @type {Array<FlatStyleRule>} */ (asArray);
    const previousFilters = [];
    for (const rule of rules) {
      /** @type {Array<FlatStyle>} */
      const ruleStyles = Array.isArray(rule.style) ? rule.style : [rule.style];
      /** @type {import("../../expr/expression.js").EncodedExpression} */
      let currentFilter = rule.filter;
      if (rule.else && previousFilters.length) {
        currentFilter = [
          'all',
          ...previousFilters.map((filter) => ['!', filter]),
        ];
        if (rule.filter) {
          currentFilter.push(rule.filter);
        }
        if (currentFilter.length < 3) {
          currentFilter = currentFilter[1];
        }
      }

      //kmg
      let featureFilter = null;
      let contextFilter = null;
      if (rule.filter) {
        previousFilters.push(rule.filter);
        //kmg
        const { featureFilters, contextFilters } = splitFilters(rule.filter);
        featureFilter = computeFeatureFilter(featureFilters);
        contextFilter = computeContextFilter(contextFilters);        
      }
      // parse each style and convert to shader
      const styleShaders = ruleStyles.map((style) =>
        ({//kmg
          ...parseLiteralStyle(style, variables, currentFilter),
          ...(featureFilter && {featureFilter}), 
          ...(contextFilter && {contextFilter}),
        })
        
      );
      shaders.push(...styleShaders);

    }
    return shaders;
  }

  // if array of shaders: return as is
  if ('builder' in asArray[0]) {
    return /** @type {Array<StyleShaders>} */ (asArray);
  }

  // array of flat styles: simply convert to shaders
  return /** @type {Array<FlatStyle>} */ (asArray).map((style) =>
    parseLiteralStyle(style, variables, null),
  );
}

//kmg
function computeFeatureFilter(filter) {
  const parsingContext = newParsingContext();
  /**
   * @type {import('../../expr/cpu.js').ExpressionEvaluator}
   */
  let compiled;
  try {
    compiled = buildExpression(filter, BooleanType, parsingContext);
  } catch {
    // filter expression failed to compile for CPU: ignore it
    return null;
  }

  // do not apply the filter if it depends on map state (e.g. zoom level) or any variable
  if (parsingContext.mapState || parsingContext.variables.size > 0) {
    return null;
  }

  const evalContext = newEvaluationContext();
  return (feature) => {
    evalContext.properties = feature.getPropertiesInternal();
    if (parsingContext.featureId) {
      const id = feature.getId();
      if (id !== undefined) {
        evalContext.featureId = id;
      } else {
        evalContext.featureId = null;
      }
    }
    evalContext.geometryType = computeGeometryType(feature.getGeometry());
    return /** @type {boolean} */ (compiled(evalContext));
  };
}

function splitFilters(filters) {
  const classifyFilters = (expression) => {
    if (!Array.isArray(expression)) return [[], []];

    const [operator, ...values] = expression;

    if (['all', 'any', '!'].includes(operator)) {
      const featureFilters = [];
      const contextFilters = [];

      values.forEach((value) => {
        const [ff, cf] = classifyFilters(value);
        featureFilters.push(...ff);
        contextFilters.push(...cf);
      });

      const minOpCount = ['all', 'any'].includes(operator) ? 1 : 0;
      return [
        featureFilters.length > minOpCount ? [[operator, ...featureFilters]] : featureFilters,
        contextFilters.length > minOpCount ? [[operator, ...contextFilters]] : contextFilters,
      ];
    }

    const isFeatureFilter = Array.isArray(expression[1]) && ['get', 'geometry-type'].includes(expression[1][0]) || operator === 'has';
    return isFeatureFilter ? [[expression], []] : [[], [expression]];
  };

  const [featureFilters, contextFilters] = classifyFilters(filters);
  return {
    featureFilters: featureFilters.length === 1 ? featureFilters[0] : featureFilters,
    contextFilters: contextFilters.length === 1 ? contextFilters[0] : contextFilters,
  };
}

function computeContextFilter(filter) {
  const parsingContext = newParsingContext();
  /**
   * @type {import('../../expr/cpu.js').ExpressionEvaluator}
   */
  let compiled;
  try {
    compiled = buildExpression(filter, BooleanType, parsingContext);
  } catch {
    // filter expression failed to compile for CPU: ignore it
    return null;
  }

  // do not apply the filter if it depends on map state (e.g. zoom level) or any variable
  //if (parsingContext.mapState || parsingContext.variables.size > 0) {
  //  return null;
  //}

  const evalContext = newEvaluationContext();
  return (resolution) => {
    evalContext.variables = parsingContext.variables;
    evalContext.resolution = resolution;
    return /** @type {boolean} */ (compiled(evalContext));
  };
}  

function pushCustomAttributesInRenderInstructionsFromFeatures(
  renderInstructions,
  customAttributes,
  entry,
  currentIndex,
  refCounter
) {
  let shift = 0;

  for (const key in customAttributes) {
    const attr = customAttributes[key];
    const value = attr.callback.call({ ref: refCounter }, entry.feature);
    const size = attr.size ?? 1;

    let first = value?.[0] ?? value;
    if (first === UNDEFINED_PROP_VALUE) {
      console.warn('The "has" operator might return false positives.'); 
    }
    if (first === undefined) {
      first = UNDEFINED_PROP_VALUE;
    } else if (first === null) {
      first = 0;
    }
    renderInstructions[currentIndex + shift++] = first;
    for (let i = 1; i < size; i++) {
      renderInstructions[currentIndex + shift++] = value[i];
    }
  }
  
  return shift;  
}


function generatePointRenderInstructionsFromFeatures(
  features,
  renderInstructions,
  customAttributes,
  transform,
) {
  let geometriesCount = 0;
  const geometryRenderEntries = new Array(features.length);

  for (let i = 0; i < features.length; i++) {
    const feature = features[i];    
    const geometry = feature.getGeometry();
    const flatCoordinates = geometry.getFlatCoordinates();
  
    geometriesCount++;
  
    const pixelCoordinates = new Array(flatCoordinates.length);
    pixelCoordinates[0] = transform[0] * flatCoordinates[0] + transform[2] * flatCoordinates[1] + transform[4];
    pixelCoordinates[1] = transform[1] * flatCoordinates[0] + transform[3] * flatCoordinates[1] + transform[5];

    geometryRenderEntries[i] = { feature, pixelCoordinates }
  }

  // here we anticipate the amount of render instructions for points:
  // 2 instructions per vertex for position (x and y)
  // + 1 instruction per vertex per custom attributes
  const totalInstructionsCount =
    (2 + getCustomAttributesSize(customAttributes)) * geometriesCount;
  if (
    !renderInstructions ||
    renderInstructions.length !== totalInstructionsCount
  ) {
    renderInstructions = new Float32Array(totalInstructionsCount);
  }    
  
  let renderIndex = 0;
  let refCounter = 0;
  for (const entry of geometryRenderEntries) {
    ++refCounter;

    renderInstructions[renderIndex++] = entry.pixelCoordinates[0];
    renderInstructions[renderIndex++] = entry.pixelCoordinates[1];

    renderIndex += pushCustomAttributesInRenderInstructionsFromFeatures(
      renderInstructions,
      customAttributes,
      entry,
      renderIndex,
      refCounter
    );   
  }
  return renderInstructions;
}

function generateLineStringRenderInstructionsFromFeatures(
  features,
  renderInstructions,
  customAttributes,
  transform,
) {
  let verticesCount = 0;
  let geometriesCount = 0;
  const geometryRenderEntries = new Array(features.length);
  
  for (let i = 0; i < features.length; i++) {
    const feature = features[i];    
    const geometry = feature.getGeometry();
    const flatCoordinates = geometry.getFlatCoordinates();
    const ends = geometry.getEnds();
    const stride = geometry.getStride();
  
    verticesCount += flatCoordinates.length / stride;
    geometriesCount += ends.length;
  
    const pixelCoordinates = new Array(flatCoordinates.length);
    transform2D(
      flatCoordinates, 
      0, 
      flatCoordinates.length, 
      stride, 
      transform, 
      pixelCoordinates, 
      stride
    );
    geometryRenderEntries[i] = { feature, pixelCoordinates, ends };
  }

  // here we anticipate the amount of render instructions for lines:
  // 3 instructions per vertex for position (x, y and m)
  // + 1 instruction per line per custom attributes
  // + 1 instruction per line (for vertices count)
  const totalInstructionsCount =
  3 * verticesCount +
  (1 + getCustomAttributesSize(customAttributes)) * geometriesCount;
  
  if (
    !renderInstructions ||
    renderInstructions.length !== totalInstructionsCount
  ) {
    renderInstructions = new Float32Array(totalInstructionsCount);
  }
  
  let renderIndex = 0;
  let refCounter = 0;
  
  for (const entry of geometryRenderEntries) {
    const stride = entry.feature.stride_;    
  
    ++refCounter;
    let offset = 0;

    const customAttrValues = [];
    const customAttrSize = pushCustomAttributesInRenderInstructionsFromFeatures(
      customAttrValues,
      customAttributes,
      entry,
      0,
      refCounter
    );
  
    for (const end of entry.ends) {
      for (let i = 0; i < customAttrSize; ++i)
        renderInstructions[renderIndex++] = customAttrValues[i];

      // vertices count
      renderInstructions[renderIndex++] = (end - offset) / stride;
  
      // looping on points for positions
      for (let i = offset; i < end; i += stride) {
        renderInstructions[renderIndex++] = entry.pixelCoordinates[i]; 
        renderInstructions[renderIndex++] = entry.pixelCoordinates[i + 1];
        renderInstructions[renderIndex++] = stride === 3 ? entry.pixelCoordinates[i + 2] : 0;
      }
      offset = end;
    }
  }  
  return renderInstructions;
}

function generatePolygonRenderInstructionsFromFeatures(
  features,
  renderInstructions,
  customAttributes,
  transform,
) {
  let verticesCount = 0;
  let geometriesCount = 0;
  let ringsCount = 0;

  const geometryRenderEntries = new Array(features.length);

  for (let i = 0; i < features.length; i++) {
    const feature = features[i];    
    const geometry = feature.getGeometry();
    const flatCoordinates = geometry.getFlatCoordinates();
    const ends = geometry.getEnds();
    const stride = geometry.getStride();

    verticesCount += flatCoordinates.length / stride;
    ringsCount += ends.length;

    const multiPolygonEnds = inflateEnds(flatCoordinates, ends);
    geometriesCount += multiPolygonEnds.length;

    const pixelCoordinates = new Array(flatCoordinates.length);
    transform2D(
      flatCoordinates, 
      0, 
      flatCoordinates.length, 
      stride, 
      transform, 
      pixelCoordinates, 
      stride
    );
    geometryRenderEntries[i] = { feature, pixelCoordinates, multiPolygonEnds };
  }

  // here we anticipate the amount of render instructions for polygons:
  // 2 instructions per vertex for position (x and y)
  // + 1 instruction per polygon per custom attributes
  // + 1 instruction per polygon (for vertices count in polygon)
  // + 1 instruction per ring (for vertices count in ring)
  const totalInstructionsCount =
    2 * verticesCount +
    (1 + getCustomAttributesSize(customAttributes)) * geometriesCount +
    ringsCount;

  if (
    !renderInstructions ||
    renderInstructions.length !== totalInstructionsCount
  ) {
    renderInstructions = new Float32Array(totalInstructionsCount);
  }

  let renderIndex = 0;
  let refCounter = 0;

  for (const entry of geometryRenderEntries) {
    const stride = entry.feature.stride_;    

    ++refCounter;
    let offset = 0;

    for (const polygonEnds of entry.multiPolygonEnds) {

      renderIndex += pushCustomAttributesInRenderInstructionsFromFeatures(
        renderInstructions,
        customAttributes,
        entry,
        renderIndex,
        refCounter
      );

      // ring count
      const ringsVerticesCount = polygonEnds.length;
      renderInstructions[renderIndex++] = ringsVerticesCount;

      // vertices count in each ring
      for (let i = 0; i < ringsVerticesCount; i++) {
        renderInstructions[renderIndex++] =
          (polygonEnds[i] - (i === 0 ? offset : polygonEnds[i - 1])) / stride;
      }

      // looping on points for positions
      for (let i = 0; i < ringsVerticesCount; i++) {
        let end = polygonEnds[i];

        for (let j = offset; j < end; j += 2) {
          renderInstructions[renderIndex++] = entry.pixelCoordinates[j];
          renderInstructions[renderIndex++] = entry.pixelCoordinates[j + 1];
        }
        offset = end;
      }
    }
  }
  return renderInstructions;
}


// Point buffer generation
function generatePointBuffers_(instructions, customAttributesSize) {
  const customAttrsCount = customAttributesSize;
  const instructionsPerElement = 2 + customAttrsCount;
  const elementCount = instructions.length / instructionsPerElement;
  
  const instanceAttributes = new Float32Array(
    elementCount * (2 + customAttrsCount)
  );

  let bufferPosition = 0;
  for (let i = 0; i < instructions.length; i += instructionsPerElement) {
    instanceAttributes[bufferPosition++] = instructions[i];     // x
    instanceAttributes[bufferPosition++] = instructions[i + 1]; // y
    
    for (let j = 0; j < customAttrsCount; j++) {
      instanceAttributes[bufferPosition++] = instructions[i + 2 + j];
    }
  }

  return {
    indicesBuffer : new Uint32Array([0, 1, 3, 1, 2, 3]),
    vertexAttributesBuffer : new Float32Array([-1, -1, 1, -1, 1, 1, -1, 1]),
    instanceAttributesBuffer : instanceAttributes,
  };

}

function generatePolygonBuffers_(instructions, customAttributesSize) {
  const customAttrsCount = customAttributesSize;  
  const instructionsPerVertex = 2; // x, y

  let currentInstructionsIndex = 0;
  let totalVertexCount = 0;
  let maxTotalIndices = 0;
  while (currentInstructionsIndex < instructions.length) {
    currentInstructionsIndex += customAttrsCount;
    const ringsCount = instructions[currentInstructionsIndex++];
    let flatCoordsCount = 0;
    for (let i = 0; i < ringsCount; ++i) {
      flatCoordsCount += instructions[currentInstructionsIndex++];
    }
    currentInstructionsIndex += flatCoordsCount * 2;
    totalVertexCount += flatCoordsCount;
    maxTotalIndices += flatCoordsCount * 3;
  }

  const vertices = new Float32Array(totalVertexCount * (2 + customAttrsCount));
  const indices = new Uint32Array(maxTotalIndices);

  let instructionsIndex = 0;
  let vertexOffset = 0;
  let indexOffset = 0;
  while (instructionsIndex < instructions.length) {
    const customAttributes = instructions.slice(
      instructionsIndex,
      instructionsIndex + customAttrsCount,
    );
    instructionsIndex += customAttrsCount;

    const ringsCount = instructions[instructionsIndex++];
    const holes = new Array(ringsCount - 1);
    let verticesCount = 0;
    for (let i = 0; i < ringsCount; ++i) {
      //const ringVertexCount = instructions[instructionsIndex++];
      verticesCount += instructions[instructionsIndex++];
      if (i < ringsCount - 1) 
        holes[i] = verticesCount;
    }

    const flatCoords = instructions.slice(
      instructionsIndex,
      instructionsIndex + verticesCount * instructionsPerVertex,
    );

    const result = earcut(flatCoords, holes, instructionsPerVertex);

    for (let i = 0; i < verticesCount; ++i) {
      const base = (vertexOffset + i) * (2 + customAttrsCount);
      vertices[base] = flatCoords[i * 2];
      vertices[base + 1] = flatCoords[i * 2 + 1];
      for (let j = 0; j < customAttrsCount; ++j) {
        vertices[base + 2 + j] = customAttributes[j];
      }
    }

    for (let i = 0; i < result.length; ++i) {
      indices[indexOffset + i] = result[i] + vertexOffset;
    }
    vertexOffset += verticesCount;
    indexOffset += result.length;

    instructionsIndex += verticesCount * instructionsPerVertex;
  }

  return {
    indicesBuffer: (indexOffset < indices.length) ? indices.slice(0, indexOffset) : indices,
    vertexAttributesBuffer: vertices,
    instanceAttributesBuffer: new Float32Array([]),
  };
}

function generateLineStringBuffers_(renderInstructions, customAttributesSize, transform) {
  const customAttrsCount = customAttributesSize;
  const instructionsPerVertex = 3;

  let currentInstructionsIndex = 0;
  let totalSegments = 0;
  while (currentInstructionsIndex < renderInstructions.length) {
    currentInstructionsIndex += customAttrsCount;
    const verticesCount = renderInstructions[currentInstructionsIndex++];
    totalSegments += (verticesCount - 1);
    currentInstructionsIndex += verticesCount * instructionsPerVertex;
  }  

  const floatsPerSegment =
    2 +                // p0(x, y)
    1 +                // m0
    2 +                // p1(x, y)
    1 +                // m1
    2 +                // angle0, angle1
    1 +                // currentLength
    1 +                // currentAngleTangentSum
    customAttrsCount;  // customAttrs
  const totalFloats = totalSegments * floatsPerSegment;
  const instanceAttributes = new Float32Array(totalFloats);
  let instanceOffset = 0;

  const invertTransform = createTransform();
  makeInverseTransform(invertTransform, transform);

  currentInstructionsIndex = 0;
  while (currentInstructionsIndex < renderInstructions.length) {
    const customAttributes = [];
    for (let i = 0; i < customAttrsCount; ++i)
      customAttributes[i] = renderInstructions[currentInstructionsIndex + i];
    currentInstructionsIndex += customAttrsCount;
    const verticesCount = renderInstructions[currentInstructionsIndex++];

    const firstInstructionsIndex = currentInstructionsIndex;
    const lastInstructionsIndex = currentInstructionsIndex + (verticesCount - 1) * instructionsPerVertex;
    const isLoop =
      renderInstructions[firstInstructionsIndex] === renderInstructions[lastInstructionsIndex] &&
      renderInstructions[firstInstructionsIndex + 1] === renderInstructions[lastInstructionsIndex + 1];

    let currentLength = 0;
    let currentAngleTangentSum = 0;

    const worldCoordCache = [];

    function getWorldCoord(index, offset) {
      if (offset === null || offset < 0 || offset >= verticesCount) return null;

      if (worldCoordCache[index]?.offset === offset) 
        return worldCoordCache[index].world;

      const segmentStartIndex = currentInstructionsIndex + offset * instructionsPerVertex;
      const world = applyTransform(invertTransform, [
        renderInstructions[segmentStartIndex],
        renderInstructions[segmentStartIndex + 1]
      ]);
      worldCoordCache[index] = { offset, world };
      return world;
    }

    for (let i = 0; i < verticesCount - 1; ++i) {
      let beforeIndex = null;
      if (i > 0) {
        beforeIndex =
          currentInstructionsIndex + (i - 1) * instructionsPerVertex;
      } else if (isLoop) {
        beforeIndex = lastInstructionsIndex - instructionsPerVertex;
      }
      let afterIndex = null;
      if (i < verticesCount - 2) {
        afterIndex =
          currentInstructionsIndex + (i + 2) * instructionsPerVertex;
      } else if (isLoop) {
        afterIndex = firstInstructionsIndex + instructionsPerVertex;
      }

      const segmentStartIndex = currentInstructionsIndex + i * instructionsPerVertex;
      const segmentEndIndex   = currentInstructionsIndex + (i + 1) * instructionsPerVertex;
      const p0 = [renderInstructions[segmentStartIndex], renderInstructions[segmentStartIndex + 1]];
      const p1 = [renderInstructions[segmentEndIndex],   renderInstructions[segmentEndIndex + 1]];
      const m0 = renderInstructions[segmentStartIndex + 2];
      const m1 = renderInstructions[segmentEndIndex + 2];

      const idx0 = (segmentStartIndex - currentInstructionsIndex) / instructionsPerVertex;
      const idx1 = (segmentEndIndex   - currentInstructionsIndex) / instructionsPerVertex;
      let idxB = null, idxA = null;
      if (beforeIndex !== null)
        idxB = (beforeIndex - currentInstructionsIndex) / instructionsPerVertex;
      if (afterIndex !== null)
        idxA = (afterIndex - currentInstructionsIndex) / instructionsPerVertex;

      const pBworld = getWorldCoord(0, idxB);
      const p0world = getWorldCoord(1, idx0);
      const p1world = getWorldCoord(2, idx1);
      const pAworld = getWorldCoord(3, idxA);
      worldCoordCache.shift();

      function angleBetween(p0, pA, pB) {
        const ax = pA[0] - p0[0], ay = pA[1] - p0[1];
        const bx = pB[0] - p0[0], by = pB[1] - p0[1];
        if ((ax * ax + ay * ay) < 1e-12 || (bx * bx + by * by) < 1e-12) return 0;
        const angle = Math.atan2(ax * by - ay * bx, ax * bx + ay * by);
        return angle < 0 ? angle + 2 * Math.PI : angle;
      }

      let angle0 = -1, angle1 = -1;
      let newAngleTangentSum = currentAngleTangentSum;
      if (beforeIndex !== null) {
        angle0 = angleBetween(p0world, p1world, pBworld);
        if (Math.cos(angle0) <= 0.985)//LINESTRING_ANGLE_COSINE_CUTOFF
          newAngleTangentSum += Math.tan((angle0 - Math.PI) / 2);
      }
      if (afterIndex !== null) {
        angle1 = angleBetween(p1world, p0world, pAworld);
        if (Math.cos(angle1) <= 0.985)//LINESTRING_ANGLE_COSINE_CUTOFF
          newAngleTangentSum += Math.tan((Math.PI - angle1) / 2);
      }

      instanceAttributes[instanceOffset++] = p0[0];
      instanceAttributes[instanceOffset++] = p0[1];
      instanceAttributes[instanceOffset++] = m0;
      instanceAttributes[instanceOffset++] = p1[0];
      instanceAttributes[instanceOffset++] = p1[1];
      instanceAttributes[instanceOffset++] = m1;
      instanceAttributes[instanceOffset++] = angle0;
      instanceAttributes[instanceOffset++] = angle1;
      instanceAttributes[instanceOffset++] = currentLength;
      instanceAttributes[instanceOffset++] = currentAngleTangentSum;
      for (let j = 0; j < customAttrsCount; ++j)
        instanceAttributes[instanceOffset++] = customAttributes[j];

      currentLength += Math.sqrt(
        (p1world[0] - p0world[0]) * (p1world[0] - p0world[0]) +
        (p1world[1] - p0world[1]) * (p1world[1] - p0world[1])
      );
      currentAngleTangentSum = newAngleTangentSum;
    }
    currentInstructionsIndex += verticesCount * instructionsPerVertex;
  }

  return {
    indicesBuffer: new Uint32Array([0, 1, 3, 1, 2, 3]),
    vertexAttributesBuffer: new Float32Array([-1, -1, 1, -1, 1, 1, -1, 1]),
    instanceAttributesBuffer: instanceAttributes,
  };
}
